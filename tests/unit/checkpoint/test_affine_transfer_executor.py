# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Execute a shard-to-shard transfer over a real transport group and check every element.

The planner's tests compare addresses against an oracle computed a different way; a plan can be
arithmetically perfect and still be unexecutable, so these tests move bytes. Each rank fills its
target buffers with NaN rather than empty, because an unwritten element then fails equality on its
own -- a hole shows up as a difference instead of as whatever the allocator happened to leave behind.

The expected value is always the same thing: what the target map extracts from the full parameter.
That is the operation the direct route replaces, so it cannot share a bug with the addressing under
test, and it makes a wrong offset and a wrong stride indistinguishable from the failure they are.

Three cases exist because the alternatives are silent:
  * a plan two ranks disagree about must fail everywhere, since a rank that refused alone would
    leave its peers waiting on a collective that can no longer complete;
  * a buffer too small for its plan must be caught before communication, not during it;
  * disjoint endpoint sets, where no process is both a reader and a writer, which is the shape of a
    real rollout handoff and the one case where a local-copy bug has nowhere to hide.
"""

import math
import os

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from deepspeed.checkpoint.affine import contiguous_split_map, replicated_map
from deepspeed.checkpoint.affine_transfer import PlanBudget, plan_transfer
from deepspeed.checkpoint.affine_transfer_executor import (AffineTransferExecutionError, StagingBudget,
                                                           TransferEndpoints, execute_transfer)
from unit.common import DistributedTest

UNWRITTEN = float('nan')

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _join_transfer_group():
    """Create the group this test transfers over, and pin this rank to its own accelerator.

    Called from the test body rather than from a fixture on purpose: DistributedTest runs fixtures in
    the supervising process, which has no rank, and spawns the body into the ranks -- where the
    rendezvous variables exist but nothing has been initialized. The repository's own distributed
    tests never notice, because they build an engine and deepspeed.initialize sets the group in
    passing; here the group is the thing under test, so it is opened explicitly.
    """
    if not dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        get_accelerator().set_device(local_rank % max(get_accelerator().device_count(), 1))
        dist.init_distributed(dist_backend='nccl', init_method='env://')
    return dist.get_rank(), dist.get_world_size()


def _device():
    """Pin each rank to its own accelerator.

    Without this every rank resolves to device 0 and a 'cross-process' transfer silently becomes two
    kernels on one card, which would pass all of these while proving nothing about the transport.
    """
    local_rank = int(os.environ.get('LOCAL_RANK', dist.get_rank() if dist.is_initialized() else 0))
    device = local_rank % max(get_accelerator().device_count(), 1)
    get_accelerator().set_device(device)
    return torch.device('cuda', device)


def _full(rows, cols, dtype):
    """Identical markers that stay distinct after fp16 or bf16 conversion."""
    values = torch.arange(rows * cols, dtype=torch.float32, device=_device()).reshape(rows, cols).to(dtype)
    flat = values.reshape(-1)
    assert torch.unique(flat).numel() == flat.numel(), f'{dtype} markers collided; this case cannot detect a swap'
    return values


def _targets(map_, endpoints, dtype):
    return {
        location: torch.full(shape, UNWRITTEN, dtype=dtype, device=_device())
        for location, shape in map_.shard_shapes.items() if endpoints.target[location] == dist.get_rank()
    }


def _assert_matches(plan_targets, target_map, full):
    for location, buffer in plan_targets.items():
        want = target_map.extract(full, location)
        assert torch.equal(buffer, want), (f'target location {location}: {int((~torch.eq(buffer, want)).sum())} '
                                           f'of {want.numel()} elements differ from the extracted shard')


def _row_to_column(world_size, tp_size=2):
    rows, cols = 8 * tp_size, 8 * tp_size
    source = contiguous_split_map((rows, cols), [rows // tp_size] * tp_size, 0)
    target = contiguous_split_map((rows, cols), [cols // world_size] * world_size, 1)
    return source, target


class TestB2Transfer(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('dtype', DTYPES, ids=['fp32', 'fp16', 'bf16'])
    def test_pure_copy_lands_every_element(self, dtype):
        _join_transfer_group()
        source, target = _row_to_column(self.world_size)
        full = _full(16, 16, dtype)
        endpoints = TransferEndpoints(source={loc: loc
                                              for loc in source.shard_shapes},
                                      target={loc: loc
                                              for loc in target.shard_shapes})
        source_buffers = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == dist.get_rank()
        }
        target_buffers = _targets(target, endpoints, dtype)

        plan = plan_transfer(target, source, PlanBudget())
        stats = execute_transfer(plan, source_buffers, target_buffers, endpoints, dtype)

        _assert_matches(target_buffers, target, full)
        assert stats.chunks > 0
        assert sum(segment.numel for segment in plan.segments) == math.prod(target.logical_shape), \
            'the plan must cover the parameter exactly once'

    def test_a_strided_source_packs_rather_than_reinterprets(self):
        _join_transfer_group()
        """Row shards are contiguous; column shards stride, and reading them as packed is the classic error."""
        source = contiguous_split_map((16, 8), [8, 8], 0)
        target = contiguous_split_map((16, 8), [2, 2, 2, 2], 1)
        dtype = torch.float32
        full = _full(16, 8, dtype)
        endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1, 2: 0, 3: 1})
        source_buffers = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == dist.get_rank()
        }
        target_buffers = _targets(target, endpoints, dtype)

        execute_transfer(plan_transfer(target, source, PlanBudget()), source_buffers, target_buffers, endpoints, dtype)

        _assert_matches(target_buffers, target, full)
        for buffer in target_buffers.values():
            assert not torch.isnan(buffer).any(), 'a target region was never written'

    def test_a_tight_budget_spans_rounds_without_losing_order(self):
        _join_transfer_group()
        """Chunked into more rounds than there are segments, so a chunk boundary has to be crossed."""
        source, target = _row_to_column(self.world_size)
        dtype = torch.float32
        full = _full(16, 16, dtype)
        endpoints = TransferEndpoints(source={loc: loc
                                              for loc in source.shard_shapes},
                                      target={loc: loc
                                              for loc in target.shard_shapes})
        source_buffers = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == dist.get_rank()
        }
        target_buffers = _targets(target, endpoints, dtype)

        stats = execute_transfer(plan_transfer(target, source, PlanBudget()), source_buffers, target_buffers,
                                 endpoints, dtype, StagingBudget(bytes_per_peer=64))

        assert stats.rounds > 1, f'a 64-byte budget should split this into several rounds, got {stats.rounds}'
        _assert_matches(target_buffers, target, full)

    def test_a_replicated_source_is_read_from_one_holder(self):
        _join_transfer_group()
        """Every rank holds the whole parameter; the plan must still name one reader per region."""
        source = replicated_map((12, 8), self.world_size)
        target = contiguous_split_map((12, 8), [12 // 2] * 2, 0)
        dtype = torch.float32
        full = _full(12, 8, dtype)
        endpoints = TransferEndpoints(source={loc: loc
                                              for loc in source.shard_shapes},
                                      target={loc: loc
                                              for loc in target.shard_shapes})
        source_buffers = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == dist.get_rank()
        }
        target_buffers = _targets(target, endpoints, dtype)

        plan = plan_transfer(target, source, PlanBudget())
        execute_transfer(plan, source_buffers, target_buffers, endpoints, dtype)

        _assert_matches(target_buffers, target, full)
        assert {segment.source_rank for segment in plan.segments} == {0}
        assert sum(segment.numel for segment in plan.segments) == full.numel()

    def test_ranks_disagreeing_about_the_schedule_refuse_everywhere(self):
        _join_transfer_group()
        """A lone refusal would strand its peers, so disagreement has to become a collective failure.

        One rank sizes its staging differently, which changes the chunk schedule and therefore the
        digest. Nobody may post communication after that, so this asserts that all ranks raise rather
        than that some rank does.
        """
        source, target = _row_to_column(self.world_size)
        dtype = torch.float32
        full = _full(16, 16, dtype)
        endpoints = TransferEndpoints(source={loc: loc
                                              for loc in source.shard_shapes},
                                      target={loc: loc
                                              for loc in target.shard_shapes})
        source_buffers = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == dist.get_rank()
        }
        target_buffers = _targets(target, endpoints, dtype)
        budget = StagingBudget(bytes_per_peer=4 << 20 if dist.get_rank() == 0 else 1 << 20)

        with pytest.raises(AffineTransferExecutionError, match='transfer schedule'):
            execute_transfer(plan_transfer(target, source, PlanBudget()), source_buffers, target_buffers, endpoints,
                             dtype, budget)

    def test_a_buffer_smaller_than_its_plan_is_refused_before_communication(self):
        _join_transfer_group()
        source, target = _row_to_column(self.world_size)
        dtype = torch.float32
        full = _full(16, 16, dtype)
        endpoints = TransferEndpoints(source={loc: loc
                                              for loc in source.shard_shapes},
                                      target={loc: loc
                                              for loc in target.shard_shapes})
        source_buffers = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == dist.get_rank()
        }
        target_buffers = _targets(target, endpoints, dtype)
        smallest = min(target_buffers)
        target_buffers[smallest] = target_buffers[smallest][:1].clone()

        with pytest.raises(AffineTransferExecutionError, match='before it began'):
            execute_transfer(plan_transfer(target, source, PlanBudget()), source_buffers, target_buffers, endpoints,
                             dtype, StagingBudget(bytes_per_peer=256))


class TestB2DisjointEndpoints(DistributedTest):
    """B2-G03: four source ranks and four target ranks with nothing in common.

    Every process is either a reader or a writer, never both, so a transfer that appears to work by
    falling back to local copies fails here. Needs eight accelerators; skipped rather than faked when
    fewer are visible.
    """
    world_size = 8

    def test_source_and_target_ranks_do_not_overlap(self):
        _join_transfer_group()
        if get_accelerator().device_count() < self.world_size:
            pytest.skip(f'needs {self.world_size} visible accelerators, found {get_accelerator().device_count()}')

        source, target = _row_to_column(4, tp_size=4)
        dtype = torch.float32
        full = _full(32, 32, dtype)
        endpoints = TransferEndpoints(source={location: location
                                              for location in source.shard_shapes},
                                      target={location: location + 4
                                              for location in target.shard_shapes})
        source_buffers = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == dist.get_rank()
        }
        target_buffers = _targets(target, endpoints, dtype)

        stats = execute_transfer(plan_transfer(target, source, PlanBudget()), source_buffers, target_buffers,
                                 endpoints, dtype)

        assert stats.local_copy_bytes == 0, 'no process holds both ends, so nothing may be copied locally'
        assert stats.received_bytes > 0
        _assert_matches(target_buffers, target, full)

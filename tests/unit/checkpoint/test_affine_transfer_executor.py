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
import hashlib
import importlib.util
import json
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from deepspeed.checkpoint.affine import contiguous_split_map, replicated_map, segmented_map
from deepspeed.checkpoint.affine_transfer import PlanBudget, plan_transfer
from deepspeed.checkpoint.autoep_transfer import make_autoep_transfer_request
from deepspeed.checkpoint.autoep_affine import autoep_placement_to_affine_map, legacy_uniform_autoep_placement_descriptor
from deepspeed.checkpoint.constants import (AFFINE_MAP, AUTOEP_EXPERT_PLACEMENT, AUTOEP_PARAM_EP_RANK,
                                            AUTOEP_PARAM_LOGICAL_SHAPE, DS_AUTOEP_UC_META)
from deepspeed.checkpoint.affine_transfer_executor import (AffineTransferExecutionError, StagingBudget,
                                                           TransferEndpoints, TransferRequest, execute_transfer,
                                                           execute_transfers)
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

    def test_bigcode_replica_choice_changes_folding_and_executes_exactly(self):
        _join_transfer_group()
        blocks = [(32, False), (16, True)]
        source = segmented_map((48, 8), blocks, 0, 2, split_widths=[[16, 16]])
        target = segmented_map((48, 8), blocks, 0, 4, split_widths=[[8, 8, 8, 8]])
        dtype = torch.float32
        full = _full(48, 8, dtype)
        endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={rank: rank // 2 for rank in target.shard_shapes})
        sources = {
            rank: source.extract(full, rank).contiguous()
            for rank in source.shard_shapes if endpoints.source[rank] == dist.get_rank()
        }

        def private_query_holder(request):
            return request.target_rank // 2

        def highest_holder(request):
            return max(candidate.source_rank for candidate in request.candidates)

        for selector, expected_segments in ((None, 7), (private_query_holder, 6), (highest_holder, 7)):
            plan = plan_transfer(target, source, replica_selector=selector)
            targets = _targets(target, endpoints, dtype)
            execute_transfer(plan, sources, targets, endpoints, dtype)
            assert len(plan.segments) == expected_segments
            assert plan.moved_elements == 768
            _assert_matches(targets, target, full)

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


class TestB2BatchTransfer(DistributedTest):
    world_size = 2

    def test_many_small_tensors_share_peer_messages(self):
        _join_transfer_group()
        source, target = _row_to_column(self.world_size)
        endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1})
        requests = []
        expected = []
        for index in range(16):
            full = _full(16, 16, torch.float32) + index * 1024
            sources = {
                location: source.extract(full, location).contiguous()
                for location in source.shard_shapes if endpoints.source[location] == dist.get_rank()
            }
            targets = _targets(target, endpoints, torch.float32)
            requests.append(TransferRequest(f'tensor-{index}', plan_transfer(target, source), sources, targets))
            expected.append((targets, full))

        stats = execute_transfers(requests, endpoints, torch.float32)
        assert stats.rounds == 1
        for targets, full in expected:
            _assert_matches(targets, target, full)

        for targets, _ in expected:
            for buffer in targets.values():
                buffer.fill_(UNWRITTEN)
        stats = execute_transfers(requests, endpoints, torch.float32, StagingBudget(bytes_per_peer=128))
        assert stats.rounds > 1
        for targets, full in expected:
            _assert_matches(targets, target, full)

    def test_batch_refusal_reaches_every_rank(self):
        _join_transfer_group()
        source, target = _row_to_column(self.world_size)
        full = _full(16, 16, torch.float32)
        endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1})
        sources = {
            location: source.extract(full, location).contiguous()
            for location in source.shard_shapes if endpoints.source[location] == dist.get_rank()
        }
        targets = _targets(target, endpoints, torch.float32)
        if dist.get_rank() == 0:
            targets[0] = targets[0][:1].clone()
        request = TransferRequest('one', plan_transfer(target, source), sources, targets)
        with pytest.raises(AffineTransferExecutionError, match='rejected the transfer batch'):
            execute_transfers([request], endpoints, torch.float32)

    def test_batch_parameter_order_must_match(self):
        _join_transfer_group()
        source, target = _row_to_column(self.world_size)
        full = _full(16, 16, torch.float32)
        endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1})
        sources = {
            location: source.extract(full, location).contiguous()
            for location in source.shard_shapes if endpoints.source[location] == dist.get_rank()
        }
        plan = plan_transfer(target, source)
        first = TransferRequest('first', plan, sources, _targets(target, endpoints, torch.float32))
        second = TransferRequest('second', plan, sources, _targets(target, endpoints, torch.float32))
        requests = [first, second] if dist.get_rank() == 0 else [second, first]
        with pytest.raises(AffineTransferExecutionError, match='batch schedule'):
            execute_transfers(requests, endpoints, torch.float32)


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


class TestB2TransferCPU:

    @pytest.mark.parametrize('shape', [(3, 7, 11), (13, 17, 1), (2, 5, 7, 3)])
    def test_nd_tiles_fit_a_small_packed_budget_and_preserve_guards(self, shape):
        # Last-axis-only splitting rejects these valid shapes even when one element fits.
        numel = math.prod(shape)
        full = torch.arange(numel, dtype=torch.float32).reshape(shape)
        source_storage = torch.full((numel + 10, ), -7.0)
        target_storage = torch.full((numel + 12, ), -9.0)
        source = source_storage[5:5 + numel].view(shape)
        target = target_storage[6:6 + numel].view(shape)
        source.copy_(full)
        map_ = replicated_map(shape, 1)
        request = TransferRequest('expert', plan_transfer(map_, map_), {0: source}, {0: target})
        stats = execute_transfers([request], TransferEndpoints({0: 0}, {0: 0}), torch.float32,
                                  StagingBudget(bytes_per_peer=28))
        assert torch.equal(target, full)
        assert torch.equal(target_storage[:6], torch.full((6, ), -9.0))
        assert torch.equal(target_storage[-6:], torch.full((6, ), -9.0))
        assert stats.local_copy_bytes == numel * 4
        assert stats.chunks > 1

    @pytest.mark.parametrize('shape', [(4, 7, 11), (4, 5, 7, 3)])
    def test_nd_strided_pieces_land_in_the_target_map(self, shape):
        source_map = contiguous_split_map(shape, [2, 2], 0)
        last = shape[-1]
        target_map = contiguous_split_map(shape, [last // 2, last - last // 2], len(shape) - 1)
        full = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
        sources = {rank: source_map.extract(full, rank).contiguous() for rank in source_map.shard_shapes}
        targets = {rank: torch.full(local_shape, -1.0) for rank, local_shape in target_map.shard_shapes.items()}
        endpoints = TransferEndpoints({0: 0, 1: 0}, {0: 0, 1: 0})
        execute_transfer(plan_transfer(target_map, source_map), sources, targets, endpoints, torch.float32,
                         StagingBudget(bytes_per_peer=20))
        for rank, target in targets.items():
            assert torch.equal(target, target_map.extract(full, rank))

    @pytest.mark.parametrize('invalid', [
        'source_noncontiguous',
        'target_noncontiguous',
        'missing_source',
        'missing_target',
        'negative_offset',
        'short_stride',
        'zero_stride',
        'wrong_dtype',
        'zero_budget',
        'sub_element_budget',
        'unowned_source',
        'unowned_target',
        'invalid_endpoint',
    ])
    @pytest.mark.parametrize('batched', [False, True])
    def test_invalid_buffers_and_metadata_are_rejected_before_any_write(self, invalid, batched):
        map_ = replicated_map((2, 3), 1)
        plan = plan_transfer(map_, map_)
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        target = torch.full((2, 3), -11.0)
        targets = {0: target}
        sources = {0: source}
        endpoints = TransferEndpoints({0: 0}, {0: 0})
        budget = StagingBudget(16)
        if invalid == 'source_noncontiguous':
            sources[0] = source.t()
        elif invalid == 'target_noncontiguous':
            targets[0] = target.t()
        elif invalid == 'missing_source':
            sources = {}
        elif invalid == 'missing_target':
            targets = {}
        elif invalid in ('negative_offset', 'short_stride', 'zero_stride'):
            changes = {
                'negative_offset': {
                    'source_offset': -1
                },
                'short_stride': {
                    'source_strides': (1, )
                },
                'zero_stride': {
                    'target_strides': (0, 1)
                }
            }
            segment = replace(plan.segments[0], **changes[invalid])
            plan = replace(plan, segments=(segment, ))
        elif invalid == 'wrong_dtype':
            sources[0] = source.to(torch.bfloat16)
        elif invalid == 'zero_budget':
            budget = StagingBudget(0)
        elif invalid == 'sub_element_budget':
            budget = StagingBudget(3)
        elif invalid == 'unowned_source':
            sources = {1: source}
        elif invalid == 'unowned_target':
            targets = {1: target}
        elif invalid == 'invalid_endpoint':
            endpoints = TransferEndpoints({0: 1}, {0: 0})
        first_target = torch.full((2, 3), -13.0)
        with pytest.raises(AffineTransferExecutionError):
            if batched:
                first = TransferRequest('first', plan_transfer(map_, map_), {0: source}, {0: first_target})
                second = TransferRequest('second', plan, sources, targets)
                execute_transfers([first, second], endpoints, torch.float32, budget)
            else:
                execute_transfer(plan, sources, targets, endpoints, torch.float32, budget)
        assert torch.equal(target, torch.full((2, 3), -11.0))
        assert torch.equal(first_target, torch.full((2, 3), -13.0))

    @pytest.mark.parametrize('alias', ['read_write', 'write_write'])
    def test_aliases_between_requests_are_rejected_before_any_write(self, alias):
        map_ = replicated_map((6, ), 1)
        plan = plan_transfer(map_, map_)
        shared = torch.full((6, ), -11.0)
        other = torch.full((6, ), -13.0)
        first = TransferRequest('first', plan, {0: torch.arange(6, dtype=torch.float32)}, {0: shared})
        second = TransferRequest('second', plan,
                                 {0: shared if alias == 'read_write' else torch.arange(6, dtype=torch.float32)},
                                 {0: other if alias == 'read_write' else shared})
        with pytest.raises(AffineTransferExecutionError):
            execute_transfers([first, second], TransferEndpoints({0: 0}, {0: 0}), torch.float32)
        assert torch.equal(shared, torch.full((6, ), -11.0))
        assert torch.equal(other, torch.full((6, ), -13.0))

    def test_disjoint_windows_of_one_storage_are_valid(self):
        map_ = replicated_map((6, ), 1)
        storage = torch.arange(12, dtype=torch.float32)
        source, target = storage[:6], storage[6:]
        expected = source.clone()
        execute_transfer(plan_transfer(map_, map_), {0: source}, {0: target}, TransferEndpoints({0: 0}, {0: 0}),
                         torch.float32, StagingBudget(8))
        assert torch.equal(target, expected)
        assert torch.equal(source, expected)

    def test_frozen_parameters_keep_their_identity_and_storage(self):
        map_ = replicated_map((2, 3), 1)
        source = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        target = torch.nn.Parameter(torch.full((2, 3), -1.0))
        identity, pointer = id(target), target.data_ptr()
        execute_transfer(plan_transfer(map_, map_), {0: source}, {0: target}, TransferEndpoints({0: 0}, {0: 0}),
                         torch.float32, StagingBudget(8))
        assert id(target) == identity and target.data_ptr() == pointer
        assert target.requires_grad and target.grad is None
        assert torch.equal(target, source)


class TestB2AutoEPTransferCPU:

    def _parameter(self, full, ep_size, location):
        shape = tuple(full.shape)
        placement = legacy_uniform_autoep_placement_descriptor(shape[0], shape[0] // ep_size, ep_size)
        map_ = autoep_placement_to_affine_map(placement, shape)
        parameter = torch.nn.Parameter(map_.extract(full, location))
        metadata = {
            AUTOEP_EXPERT_PLACEMENT: placement,
            AFFINE_MAP: map_.to_dict(),
            AUTOEP_PARAM_LOGICAL_SHAPE: shape,
            AUTOEP_PARAM_EP_RANK: location
        }
        setattr(parameter, DS_AUTOEP_UC_META, metadata)
        return parameter, metadata

    def test_metadata_drives_expert_repartition_into_existing_parameters(self):
        full = torch.arange(8 * 15, dtype=torch.float32).reshape(8, 3, 5)
        source, source_metadata = self._parameter(full, 1, 0)
        targets = {rank: self._parameter(full, 2, rank)[0] for rank in range(2)}
        target_metadata = getattr(targets[0], DS_AUTOEP_UC_META)
        with torch.no_grad():
            for parameter in targets.values():
                parameter.fill_(-1)
        pointers = {rank: parameter.data_ptr() for rank, parameter in targets.items()}
        request = make_autoep_transfer_request('layer.expert_projection', source_metadata, target_metadata,
                                               {0: source}, targets)
        execute_transfers([request], TransferEndpoints({0: 0}, {0: 0, 1: 0}), torch.float32, StagingBudget(28))
        assert request.preflight_error is None
        for rank, parameter in targets.items():
            assert parameter.data_ptr() == pointers[rank]
            assert torch.equal(parameter, full[rank * 4:(rank + 1) * 4])

    @pytest.mark.parametrize('invalid', ['missing_metadata', 'wrong_rank', 'wrong_shape', 'fragment'])
    def test_adapter_errors_reject_the_whole_batch(self, invalid):
        full = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
        source, source_metadata = self._parameter(full, 1, 0)
        target, target_metadata = self._parameter(full, 1, 0)
        with torch.no_grad():
            target.fill_(-11)
        bad_source, bad_metadata = self._parameter(full, 1, 0)
        if invalid == 'missing_metadata':
            delattr(bad_source, DS_AUTOEP_UC_META)
        elif invalid == 'wrong_rank':
            bad_metadata[AUTOEP_PARAM_EP_RANK] = 1
        elif invalid == 'wrong_shape':
            bad_metadata[AUTOEP_PARAM_LOGICAL_SHAPE] = (4, 6)
        elif invalid == 'fragment':
            with torch.no_grad():
                bad_source.data = bad_source.data.view(-1)[:3]
        first = make_autoep_transfer_request('first', source_metadata, target_metadata, {0: source}, {0: target})
        bad_target, bad_target_metadata = self._parameter(full, 1, 0)
        with torch.no_grad():
            bad_target.fill_(-13)
        second = make_autoep_transfer_request('second', source_metadata, bad_target_metadata, {0: bad_source},
                                              {0: bad_target})
        assert second.preflight_error is not None
        with pytest.raises(AffineTransferExecutionError):
            execute_transfers([first, second], TransferEndpoints({0: 0}, {0: 0}), torch.float32)
        assert torch.equal(target, torch.full_like(target, -11))
        assert torch.equal(bad_target, torch.full_like(bad_target, -13))


class TestB2GateCPU:

    @pytest.mark.parametrize('fault', ['duplicate_cases', 'changed_source', 'omitted_source', 'failed_status'])
    def test_consumer_refuses_incomplete_or_stale_correctness_evidence(self, tmp_path, fault):
        # A copied PASS flag must not admit omitted cases or a different executor.
        root = Path(__file__).resolve().parents[3]
        path = root / 'tests/distributed/checkpoint/affine_handoff_h100.py'
        spec = importlib.util.spec_from_file_location('affine_handoff_gate_contract', path)
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        sources = ('deepspeed/checkpoint/affine_transfer.py', 'deepspeed/checkpoint/affine_transfer_executor.py',
                   'deepspeed/checkpoint/autoep_transfer.py', 'tests/distributed/checkpoint/affine_handoff_h100.py')
        manifest = {
            'world_size': 8,
            'scope': 'cross-node',
            'cases': [f'U{index}' for index in range(1, 9)],
            'source_sha256': {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in sources
            }
        }
        result = {'status': 'PASS', 'eight_gpu_correctness_gate': True}
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
        (tmp_path / 'correctness.json').write_text(json.dumps(result))
        harness.require_passed_gate(tmp_path)
        if fault == 'duplicate_cases':
            manifest['cases'] = ['U1'] * 8
        elif fault == 'changed_source':
            manifest['source_sha256'][sources[1]] = '0' * 64
        elif fault == 'omitted_source':
            manifest['source_sha256'].pop(sources[1])
        else:
            result['status'] = 'FAIL'
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
        (tmp_path / 'correctness.json').write_text(json.dumps(result))
        with pytest.raises(ValueError):
            harness.require_passed_gate(tmp_path)


class TestB2TimingCPU:

    def test_delayed_pilot_does_not_shorten_the_other_paired_route(self):
        root = Path(__file__).resolve().parents[3]
        path = root / 'benchmarks/checkpoint/affine_handoff_h100.py'
        spec = importlib.util.spec_from_file_location('affine_handoff_paired_window_contract', path)
        benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark)
        # A slow pilot on one route must not admit a brief window on the faster route.
        pilots, minimum = (0.02, 0.5), 60
        iterations = benchmark.paired_iteration_count(pilots, minimum)
        assert all(iterations * duration >= minimum for duration in pilots)


class TestB2ModelMoveCPU:

    def test_runtime_placement_survives_parameter_replacement_during_model_move(self):
        root = Path(__file__).resolve().parents[3]
        path = root / 'tests/distributed/checkpoint/autoep_handoff_e2e_h100.py'
        spec = importlib.util.spec_from_file_location('autoep_handoff_model_move_contract', path)
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        full = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
        parameter, metadata = TestB2AutoEPTransferCPU()._parameter(full, 2, 0)
        model = torch.nn.Module()
        model.register_parameter('expert', parameter)
        # CPU-to-meta conversion exercises real Parameter replacement without a GPU import stub.
        harness.move_model_preserving_expert_metadata(model, torch.device('meta'))
        assert model.expert.is_meta
        assert tuple(model.expert.shape) == (2, 3, 2)
        assert getattr(model.expert, DS_AUTOEP_UC_META) == metadata


class TestB2GenerationCPU:

    def test_consumer_uses_its_own_greedy_tokens_and_retains_the_cache(self):
        root = Path(__file__).resolve().parents[3]
        path = root / 'tests/distributed/checkpoint/autoep_handoff_e2e_h100.py'
        spec = importlib.util.spec_from_file_location('autoep_handoff_generation_contract', path)
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)

        class IncrementingCausalLM(torch.nn.Module):

            def forward(self, input_ids, attention_mask, position_ids, past_key_values=None, use_cache=True):
                # This collaborator follows the public causal-LM/cache interface.
                previous = 0 if past_key_values is None else past_key_values.length
                length = previous + input_ids.shape[1]
                assert attention_mask.shape[1] == length
                assert use_cache
                logits = torch.zeros(*input_ids.shape, 16)
                next_token = (input_ids[:, -1] + 1) % 16
                logits[:, -1].scatter_(1, next_token[:, None], 1)
                return SimpleNamespace(logits=logits, past_key_values=SimpleNamespace(length=length))

        trace = [{'input_ids': torch.tensor([[2, 3], [4, 5]]), 'attention_mask': torch.ones(2, 2)}]
        # Later oracle inputs deliberately differ; replaying them would hide a broken generation loop.
        trace.extend({
            'input_ids': torch.full((2, 1), 12),
            'attention_mask': torch.ones(2, length)
        } for length in (3, 4))
        _, durations, tokens = harness.consume(IncrementingCausalLM(), trace, {}, torch.device('cpu'), False)
        assert torch.equal(tokens, torch.tensor([[4, 6], [5, 7], [6, 8]]))
        assert len(durations) == 3

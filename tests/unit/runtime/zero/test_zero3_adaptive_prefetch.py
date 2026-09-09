# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Adaptive prefetch bounds, trace lifecycle, timing, and distributed agreement."""

from unittest.mock import Mock, patch

import pytest
import torch

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partitioned_param_coordinator import (
    InflightParamRegistry,
    PartitionedParameterCoordinator,
    ZeRoTraceMode,
)
from unit.common import DistributedTest

PREFIX = "_PartitionedParameterCoordinator__"
MODULE = "deepspeed.runtime.zero.partitioned_param_coordinator"


def _make_coordinator(prefetch_bucket_sz=50_000_000,
                      adaptive_prefetch=True,
                      adaptive_prefetch_min_sz=10_000_000,
                      adaptive_prefetch_max_sz=500_000_000,
                      max_available_parameters_in_numel=1_000_000_000,
                      dp_process_group=None):
    return PartitionedParameterCoordinator(
        prefetch_bucket_sz=prefetch_bucket_sz,
        max_reuse_distance_in_numel=int(1e9),
        max_available_parameters_in_numel=max_available_parameters_in_numel,
        allgather_stream=get_accelerator().default_stream(),
        inflight_param_registry=InflightParamRegistry(),
        adaptive_prefetch=adaptive_prefetch,
        adaptive_prefetch_min_sz=adaptive_prefetch_min_sz,
        adaptive_prefetch_max_sz=adaptive_prefetch_max_sz,
        dp_process_group=dp_process_group,
    )


def _bucket(coordinator):
    return getattr(coordinator, PREFIX + "prefetch_bucket_sz")


def _complete_trace(coordinator):
    setattr(coordinator, PREFIX + "trace_mode", ZeRoTraceMode.COMPLETE)
    coordinator.reset_step()


def _sample(coordinator, wait_ratio, forward=True):
    # Model a 10 ms interval with a known communication stall. The next reset_step
    # consumes these observations through the production iteration-boundary path.
    events = getattr(coordinator, PREFIX + "adaptive_fetch_events")
    events.append((0.0, 0.01 * (1.0 - wait_ratio), 0.01, forward))


def _drive_adaptation(coordinator, wait_ratio, n_steps=10):
    with patch.object(get_accelerator(), "use_host_timers", return_value=True):
        for _ in range(n_steps):
            if getattr(coordinator, PREFIX + "adaptive_sample_step"):
                _sample(coordinator, wait_ratio)
            coordinator.reset_step()


class TestAdaptivePrefetchConfig:

    @pytest.mark.parametrize("initial, expected", [(0, 10_000_000), (1_000_000_000, 500_000_000)])
    def test_initial_bucket_respects_bounds(self, initial, expected):
        coord = _make_coordinator(prefetch_bucket_sz=initial)
        assert _bucket(coord) == expected

    @pytest.mark.parametrize("initial", [0, 1_000_000_000])
    def test_disabled_preserves_static_bucket(self, initial):
        coord = _make_coordinator(prefetch_bucket_sz=initial, adaptive_prefetch=False)
        _complete_trace(coord)
        with patch(MODULE + ".time.perf_counter", side_effect=AssertionError("unexpected timer")):
            _drive_adaptation(coord, 0.9, n_steps=30)
        assert _bucket(coord) == initial

    @pytest.mark.parametrize("budget", [0, 100, 100_000_000])
    def test_live_parameter_budget_caps_bounds(self, budget):
        coord = _make_coordinator(prefetch_bucket_sz=1_000_000_000, max_available_parameters_in_numel=budget)
        assert _bucket(coord) == budget
        _complete_trace(coord)
        _drive_adaptation(coord, 0.9, n_steps=30)
        assert _bucket(coord) == budget

    @pytest.mark.parametrize("minimum, maximum", [(-1, 10), (20, 10)])
    def test_rejects_invalid_bounds(self, minimum, maximum):
        with pytest.raises(ValueError, match="adaptive_prefetch_min_sz"):
            _make_coordinator(adaptive_prefetch_min_sz=minimum, adaptive_prefetch_max_sz=maximum)


class TestAdaptivePrefetchLogic:

    @pytest.mark.parametrize("ratio, expected", [(0.3, 62_500_000), (0.01, 45_000_000), (0.09, 50_000_000)])
    def test_resizes_only_at_iteration_boundary(self, ratio, expected):
        coord = _make_coordinator()
        _complete_trace(coord)
        _drive_adaptation(coord, ratio, n_steps=9)
        assert _bucket(coord) == 50_000_000
        _drive_adaptation(coord, ratio, n_steps=1)
        assert _bucket(coord) == expected

    @pytest.mark.parametrize("ratio, expected", [(0.99, 60_000_000), (0.001, 40_000_000)])
    def test_repeated_updates_respect_bounds(self, ratio, expected):
        coord = _make_coordinator(adaptive_prefetch_min_sz=40_000_000, adaptive_prefetch_max_sz=60_000_000)
        _complete_trace(coord)
        _drive_adaptation(coord, ratio, n_steps=100)
        assert _bucket(coord) == expected

    def test_zero_window_can_grow(self):
        coord = _make_coordinator(prefetch_bucket_sz=0, adaptive_prefetch_min_sz=0)
        _complete_trace(coord)
        _drive_adaptation(coord, 0.9)
        assert _bucket(coord) == 1

    def test_backward_stalls_are_not_diluted_by_forward(self):
        coord = _make_coordinator()
        _complete_trace(coord)
        _drive_adaptation(coord, 0.0, n_steps=9)
        _sample(coord, 0.01, forward=True)
        _sample(coord, 0.9, forward=False)
        with patch.object(get_accelerator(), "use_host_timers", return_value=True):
            coord.reset_step()
        assert _bucket(coord) > 50_000_000

    def test_invalidation_discards_timings_without_changing_collective_cadence(self):
        coord = _make_coordinator()
        _complete_trace(coord)
        _drive_adaptation(coord, 0.9, n_steps=9)
        _sample(coord, 0.9)
        count = getattr(coord, PREFIX + "adaptive_step_count")
        coord._invalidate_trace()
        assert not getattr(coord, PREFIX + "adaptive_fetch_events")
        assert getattr(coord, PREFIX + "adaptive_previous_fetch") is None
        assert getattr(coord, PREFIX + "adaptive_step_count") == count
        assert getattr(coord, PREFIX + "adaptive_sample_step")
        with patch(MODULE + ".assert_ints_same_as_other_ranks"):
            coord.reset_step()
        assert _bucket(coord) == 50_000_000

    @pytest.mark.parametrize("mode", [ZeRoTraceMode.INVALID, ZeRoTraceMode.RECORD, ZeRoTraceMode.COMPLETE])
    @pytest.mark.parametrize("sample_step", [False, True])
    def test_fetch_records_only_completed_trace_samples(self, mode, sample_step):
        coord = _make_coordinator()
        setattr(coord, PREFIX + "trace_mode", mode)
        setattr(coord, PREFIX + "adaptive_sample_step", sample_step)
        module = torch.nn.Module()
        module.ds_id = 0
        module.ds_external_parameters = lambda: iter(())
        with patch.object(get_accelerator(), "use_host_timers", return_value=True), \
                patch(MODULE + ".time.perf_counter", side_effect=[0.0, 0.01, 0.03, 0.04]) as timer:
            coord._fetch_sub_module_impl(module, forward=True, is_leaf=False)
            coord._fetch_sub_module_impl(module, forward=True, is_leaf=False)
        should_record = sample_step and mode == ZeRoTraceMode.COMPLETE
        assert timer.call_count == (4 if should_record else 0)
        assert len(getattr(coord, PREFIX + "adaptive_fetch_events")) == int(should_record)
        assert _bucket(coord) == 50_000_000

    def test_device_events_measure_stalls_without_fetch_synchronization(self):
        coord = _make_coordinator()
        _complete_trace(coord)
        _drive_adaptation(coord, 0.0, n_steps=9)
        module = torch.nn.Module()
        module.ds_id = 0
        module.ds_external_parameters = lambda: iter(())
        events = [Mock() for _ in range(4)]
        events[1].elapsed_time.return_value = 10.0
        events[2].elapsed_time.return_value = 8.0
        with patch.object(get_accelerator(), "use_host_timers", return_value=False), \
                patch.object(type(get_accelerator()), "Event", side_effect=events), \
                patch(MODULE + ".time.perf_counter", side_effect=AssertionError("unexpected host timer")):
            coord._fetch_sub_module_impl(module, forward=True, is_leaf=False)
            coord._fetch_sub_module_impl(module, forward=True, is_leaf=False)
            for event in events:
                event.synchronize.assert_not_called()
            coord.reset_step()
        events[3].synchronize.assert_called_once_with()
        events[2].elapsed_time.assert_called_once_with(events[3])
        assert _bucket(coord) == 62_500_000


class TestAdaptivePrefetchDistributed(DistributedTest):
    world_size = 2
    requires_cuda_env = False

    def test_different_rank_timings_choose_identical_windows(self):
        coord = _make_coordinator(dp_process_group=dist.get_world_group())
        _complete_trace(coord)
        _drive_adaptation(coord, 0.01 if dist.get_rank() == 0 else 0.9)
        sizes = [torch.empty(1, dtype=torch.int64, device=get_accelerator().current_device_name()) for _ in range(2)]
        size = torch.tensor([_bucket(coord)], dtype=torch.int64, device=get_accelerator().current_device_name())
        dist.all_gather(sizes, size)
        assert [value.item() for value in sizes] == [62_500_000, 62_500_000]

    def test_invalid_sample_on_one_rank_skips_update_on_all_ranks(self):
        coord = _make_coordinator(dp_process_group=dist.get_world_group())
        _complete_trace(coord)
        _drive_adaptation(coord, 0.9, n_steps=9)
        if dist.get_rank() == 0:
            _sample(coord, 0.9)
        with patch.object(get_accelerator(), "use_host_timers", return_value=True):
            coord.reset_step()
        assert _bucket(coord) == 50_000_000

    def test_subgroups_do_not_share_controller_updates(self):
        groups = [dist.new_group(ranks=[rank]) for rank in range(2)]
        coord = _make_coordinator(dp_process_group=groups[dist.get_rank()])
        _complete_trace(coord)
        _drive_adaptation(coord, 0.01 if dist.get_rank() == 0 else 0.9)
        assert _bucket(coord) == (45_000_000 if dist.get_rank() == 0 else 62_500_000)


class TestAdaptivePrefetchTraining(DistributedTest):
    requires_cuda_env = False

    @pytest.mark.parametrize("world_size", [1, 2])
    @pytest.mark.parametrize("overlap_comm", [False, True])
    def test_training_matches_unpartitioned_model(self, world_size, overlap_comm):
        import copy
        import deepspeed

        torch.manual_seed(1234)
        device = get_accelerator().current_device_name()
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8), torch.nn.ReLU(),
                                    torch.nn.Linear(8, 4)).to(device)
        reference = copy.deepcopy(model)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": overlap_comm,
                "reduce_bucket_size": 64,
                "stage3_param_persistence_threshold": 0,
                "stage3_max_reuse_distance": 0,
                "stage3_prefetch_bucket_size": 64,
                "stage3_adaptive_prefetch_bucket_size": True,
                "stage3_adaptive_prefetch_min_size": 8,
                "stage3_adaptive_prefetch_max_size": 256,
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                                               config=config)
        coord = engine.optimizer.parameter_offload.get_param_coordinator()
        assert getattr(coord, PREFIX + "adaptive_prefetch")
        assert getattr(coord, PREFIX + "adaptive_prefetch_group") is engine.seq_data_parallel_group
        for step in range(22):
            inputs = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8) / (16 + step)
            expected = reference(inputs)
            actual = engine(inputs)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            loss = actual.square().mean()
            engine.backward(loss)
            engine.step()
            expected.square().mean().backward()
            reference_optimizer.step()
            reference_optimizer.zero_grad()
            assert 8 <= _bucket(coord) <= 256
            sizes = [torch.empty(1, dtype=torch.int64, device=device) for _ in range(world_size)]
            dist.all_gather(sizes, torch.tensor([_bucket(coord)], dtype=torch.int64, device=device))
            assert len({size.item() for size in sizes}) == 1
        assert getattr(coord, PREFIX + "adaptive_wait_ratio_ema") is not None
        with deepspeed.zero.GatheredParameters(list(engine.module.parameters())):
            for actual, expected in zip(engine.module.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        engine.destroy()

    @pytest.mark.parametrize("config_format", ["dict", "file"])
    def test_inference_reads_config_and_uses_data_parallel_group(self, config_format, tmpdir):
        import json
        import deepspeed

        config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_adaptive_prefetch_bucket_size": True,
                "stage3_adaptive_prefetch_min_size": 8,
                "stage3_adaptive_prefetch_max_size": 64,
            },
        }
        if config_format == "file":
            config_path = str(tmpdir / "adaptive.json")
            if dist.get_rank() == 0:
                with open(config_path, "w") as config_file:
                    json.dump(config, config_file)
            dist.barrier()
            config = config_path
        model = torch.nn.Linear(8, 4)
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)
        coord = engine.optimizer.get_param_coordinator()
        assert getattr(coord, PREFIX + "adaptive_prefetch")
        assert getattr(coord, PREFIX + "adaptive_prefetch_group") is engine.seq_data_parallel_group
        assert _bucket(coord) == 64
        engine.destroy()

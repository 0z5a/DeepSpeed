# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Paired, max-rank timing for normalized synthetic expert-weight handoff."""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
import types
import zlib
from pathlib import Path

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine_transfer import plan_transfer
from deepspeed.checkpoint.affine_transfer_executor import StagingBudget, TransferEndpoints, TransferRequest, execute_transfers, plan_digest
from deepspeed.checkpoint.autoep_affine import autoep_placement_to_affine_map, legacy_uniform_autoep_placement_descriptor

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / 'tests/distributed/checkpoint/affine_handoff_h100.py'


def load_harness():
    spec = importlib.util.spec_from_file_location('affine_h100_correctness_harness', HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_transport(operations, group=None):
    ops = [
        dist.P2POp(dist.irecv if direction == 'recv' else dist.isend,
                   tensor,
                   dist.get_global_rank(group, peer) if group is not None else peer,
                   group=group) for direction, tensor, peer in operations
    ]
    return dist.batch_isend_irecv(ops) if ops else []


def load_legacy(path, expected_sha):
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_sha:
        raise ValueError('legacy executor source does not match the declared immutable SHA256')
    source = original.decode()
    if source.count('dist.batch_p2p(') != 2:
        raise ValueError('legacy transport adaptation does not match the published executor')
    adapted = source.replace('dist.batch_p2p(', 'legacy_transport(')
    name = 'deepspeed.checkpoint.affine_transfer_executor_legacy_benchmark'
    module = types.ModuleType(name)
    module.__package__ = 'deepspeed.checkpoint'
    module.legacy_transport = legacy_transport
    sys.modules[name] = module
    exec(compile(adapted, str(path), 'exec'), module.__dict__)
    return module, {
        'source_sha256': expected_sha,
        'adapted_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
        'adaptation': 'only port batch_p2p calls to actual current deepspeed.comm P2POp/batch_isend_irecv'
    }


class Workload:

    def __init__(self, logical_bytes, parameters, dtype, device, peer_bytes):
        self.world = dist.get_world_size()
        self.ep_size = self.world // 2
        self.rank = dist.get_rank()
        self.dtype = dtype
        self.device = device
        self.budget = StagingBudget(peer_bytes)
        self.element_size = torch.empty((), dtype=dtype).element_size()
        experts, hidden = self.world * 16, 256
        ffn = max(1, logical_bytes // (parameters * experts * hidden * self.element_size))
        self.shape = (experts, ffn, hidden)
        self.local_shape = (experts // self.ep_size, ffn, hidden)
        self.local_elements = math.prod(self.local_shape)
        self.full_elements = math.prod(self.shape)
        self.actual_logical_bytes = parameters * self.full_elements * self.element_size
        placement = legacy_uniform_autoep_placement_descriptor(experts, experts // self.ep_size, self.ep_size)
        map_ = autoep_placement_to_affine_map(placement, self.shape)
        self.plan = plan_transfer(map_, map_)
        self.endpoints = TransferEndpoints(dict(enumerate(range(self.ep_size))),
                                           dict(enumerate(range(self.ep_size, self.world))))
        self.source_group = dist.new_group(ranks=list(range(self.ep_size)))
        location = self.rank if self.rank < self.ep_size else self.rank - self.ep_size
        self.sources, self.targets, self.expected, self.requests = [], [], [], []
        for item in range(parameters):
            indices = torch.arange(self.local_elements, dtype=torch.int64, device=device)
            values = ((indices + location * self.local_elements + item * self.full_elements) % 251).to(dtype)
            if self.rank < self.ep_size:
                source, target = values.view(self.local_shape), None
                self.sources.append(source)
            else:
                source, target = None, torch.full(self.local_shape, -1, dtype=dtype, device=device)
                self.targets.append(target)
                self.expected.append(values.view(self.local_shape))
            self.requests.append(
                TransferRequest(f'expert.{item}', self.plan, {location: source} if source is not None else {},
                                {location: target} if target is not None else {}))
        self.full_arena = torch.empty(parameters * self.full_elements, dtype=dtype, device=device)
        self.source_arena = None
        self.gathered = []
        if self.sources:
            self.source_arena = torch.empty(parameters * self.local_elements, dtype=dtype, device=device)
            self.gathered = [torch.empty_like(self.source_arena) for _ in range(self.ep_size)]
        frame = (str(dtype), peer_bytes,
                 tuple((request.key, plan_digest(request.plan, self.endpoints, self.budget, self.element_size))
                       for request in self.requests))
        self.digest = zlib.crc32(repr(frame).encode())
        get_accelerator().synchronize()

    def update_marker(self, modulus):
        location = self.rank if self.rank < self.ep_size else self.rank - self.ep_size
        for item in range(len(self.requests)):
            indices = torch.arange(self.local_elements, dtype=torch.int64, device=self.device)
            values = ((indices + location * self.local_elements + item * self.full_elements) % modulus).to(self.dtype)
            if self.sources:
                self.sources[item].copy_(values.view(self.local_shape))
            else:
                self.expected[item].copy_(values.view(self.local_shape))
        get_accelerator().synchronize()

    def validate(self):
        for target, expected in zip(self.targets, self.expected):
            if not torch.equal(target, expected):
                raise AssertionError('target expert tensor differs from the same frozen source weights')

    def reset_targets(self):
        for target in self.targets:
            target.fill_(-1)
        get_accelerator().synchronize()

    def agreement(self):
        buffers = self.sources + self.targets
        rejected = any(buffer.dtype != self.dtype or tuple(buffer.shape) != self.local_shape
                       or not buffer.is_contiguous() or buffer.device != self.device for buffer in buffers)
        local = torch.tensor([self.digest, int(rejected)], dtype=torch.int64, device=self.device)
        seen = [torch.empty_like(local) for _ in range(self.world)]
        dist.all_gather(seen, local)
        if any(int(row[1].item()) for row in seen) or len({int(row[0].item()) for row in seen}) != 1:
            raise ValueError('rebuild reference rejected the frozen buffer/schedule contract')

    def rebuild_extract(self):
        self.agreement()
        if self.sources:
            for item, source in enumerate(self.sources):
                self.source_arena[item * self.local_elements:(item + 1) * self.local_elements].copy_(source.view(-1))
            dist.all_gather(self.gathered, self.source_arena, group=self.source_group)
        if self.rank == 0:
            for item in range(len(self.requests)):
                full = self.full_arena[item * self.full_elements:(item + 1) * self.full_elements].view(self.shape)
                for location, gathered in enumerate(self.gathered):
                    shard = gathered[item * self.local_elements:(item + 1) * self.local_elements].view(
                        self.local_shape)
                    full[location * self.local_shape[0]:(location + 1) * self.local_shape[0]].copy_(shard)
        dist.broadcast(self.full_arena, src=0)
        if self.targets:
            location = self.rank - self.ep_size
            for item, target in enumerate(self.targets):
                full = self.full_arena[item * self.full_elements:(item + 1) * self.full_elements].view(self.shape)
                target.copy_(full[location * self.local_shape[0]:(location + 1) * self.local_shape[0]])
        get_accelerator().current_stream().synchronize()

    def direct(self):
        execute_transfers(self.requests, self.endpoints, self.dtype, self.budget)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def paired_iteration_count(pilot_seconds, window_seconds):
    if not pilot_seconds or any(not math.isfinite(value) or value <= 0 for value in pilot_seconds):
        raise ValueError('paired pilots must have finite positive completion times')
    return max(8, math.ceil(window_seconds * 1.10 / min(pilot_seconds)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--correctness-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--logical-mib', type=int, choices=(256, 1024, 4096), default=1024)
    parser.add_argument('--parameters', type=int, choices=(1, 32, 256), default=32)
    parser.add_argument('--peer-mib', type=int, choices=(1, 8, 32), default=8)
    parser.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--windows', type=int, default=5)
    parser.add_argument('--window-seconds', type=int, default=60)
    parser.add_argument('--legacy-executor', type=Path)
    parser.add_argument('--legacy-sha256')
    args = parser.parse_args()
    if args.windows < 5 or args.window_seconds < 60:
        raise ValueError('use at least five independent paired windows of at least one minute')
    harness = load_harness()
    harness.require_passed_gate(args.correctness_dir)
    device, identities = harness.initialize(120)
    source_hosts = {row['host'] for row in identities if row['rank'] < 4}
    target_hosts = {row['host'] for row in identities if row['rank'] >= 4}
    if (dist.get_world_size() != 8 or len(source_hosts) != 1 or len(target_hosts) != 1
            or not source_hosts.isdisjoint(target_hosts)):
        raise ValueError('the paired benchmark requires two nodes with four H100s each')
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    workload = Workload(args.logical_mib << 20, args.parameters, dtype, device, args.peer_mib << 20)
    args.output.mkdir(parents=True, exist_ok=True)
    routes = {'rebuild_extract': workload.rebuild_extract, 'candidate': workload.direct}
    legacy_metadata, unsupported = None, {}
    if args.legacy_executor:
        if not args.legacy_sha256:
            raise ValueError('an immutable legacy source SHA256 is required')
        legacy, legacy_metadata = load_legacy(args.legacy_executor, args.legacy_sha256)

        def old_direct():
            legacy.execute_transfers(workload.requests, workload.endpoints, dtype,
                                     legacy.StagingBudget(args.peer_mib << 20))
            get_accelerator().current_stream().synchronize()

        routes['published_b2_transport_port'] = old_direct
    else:
        unsupported['published_b2'] = 'NOT_RUN_NO_LEGACY_SOURCE'
    # Multiple exact BF16 patterns distinguish every address across parameters before timing.
    for modulus in harness.marker_moduli(workload.full_elements * len(workload.requests)):
        workload.update_marker(modulus)
        for name, route in list(routes.items()):
            workload.reset_targets()
            try:
                route()
            except ValueError as error:
                if name != 'published_b2_transport_port':
                    raise
                unsupported[name] = {'status': 'UNSUPPORTED_OR_REJECTED', 'error': str(error)}
                del routes[name]
                continue
            workload.validate()
    workload.update_marker(251)
    for route in routes.values():
        for _ in range(8):
            route()
    free_bytes, total_bytes = torch.cuda.mem_get_info()  #ignore-cuda: whole-device free memory has no accelerator API
    capacities = harness.gather({'free_bytes': free_bytes, 'total_bytes': total_bytes})
    if any(row['free_bytes'] < 0.15 * row['total_bytes'] for row in capacities):
        raise RuntimeError('the combined resident benchmark buffers leave less than 15 percent GPU headroom')
    records = []
    for window in range(args.windows):
        order = list(routes) if window % 2 == 0 else list(reversed(routes))
        pilot_seconds = []
        for name in order:
            pilot_start = time.perf_counter()
            routes[name]()
            pilot_seconds.append(max(harness.gather(time.perf_counter() - pilot_start)))
        iterations = paired_iteration_count(pilot_seconds, args.window_seconds)
        for name in order:
            route = routes[name]
            get_accelerator().reset_peak_memory_stats()
            durations = []
            started = time.perf_counter()
            for _ in range(iterations):
                sample_start = time.perf_counter()
                route()
                durations.append(time.perf_counter() - sample_start)
            elapsed = time.perf_counter() - started
            workload.validate()
            elapsed_per_rank = harness.gather(elapsed)
            complete_window = max(elapsed_per_rank) >= args.window_seconds
            raw = harness.gather(durations)
            samples = [max(row[index] for row in raw) for index in range(iterations)]
            memories = harness.gather({
                'peak_allocated': get_accelerator().max_memory_allocated(),
                'peak_reserved': get_accelerator().max_memory_reserved()
            })
            row = {
                'window': window,
                'route': name,
                'iterations': iterations,
                'window_status': 'PASS' if complete_window else 'SHORT_WINDOW',
                'wall_seconds_per_rank': elapsed_per_rank,
                'rank_durations_seconds': raw,
                'max_rank_seconds': samples,
                'p50_seconds': statistics.median(samples),
                'p95_seconds': percentile(samples, 0.95),
                'memory': memories,
                'scope': 'normalized frozen expert tensors ready; no model/serving cycle'
            }
            if dist.get_rank() == 0:
                records.append(row)
                with (args.output / 'timings.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
            if not complete_window:
                raise RuntimeError('the measured paired window did not reach its frozen minimum duration')
    if dist.get_rank() == 0:
        result = {
            'status': 'PASS',
            'records': records,
            'unsupported': unsupported,
            'legacy': legacy_metadata,
            'actual_logical_bytes': workload.actual_logical_bytes,
            'shape': workload.shape,
            'parameters': args.parameters,
            'peer_bytes': args.peer_mib << 20,
            'dtype': args.dtype,
            'devices': identities,
            'synthetic_expert_parameters': True,
            'consumer_cycle': 'NOT_RUN',
            'expansion_admitted': False,
            'memory_scope':
            'process-resident baseline/candidate buffers and PyTorch cache; add whole-GPU telemetry separately',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'warm_capacity': capacities
        }
        (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

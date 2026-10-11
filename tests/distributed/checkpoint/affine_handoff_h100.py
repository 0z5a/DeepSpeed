# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Finite externally launched correctness checks for affine handoff and a real AutoEP consumer."""

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import socket
import subprocess
import time
import traceback
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine import contiguous_split_map, replicated_map
from deepspeed.checkpoint.affine_transfer import plan_transfer
from deepspeed.checkpoint.affine_transfer_executor import (AffineTransferExecutionError, StagingBudget,
                                                           TransferEndpoints, TransferRequest, execute_transfers)
from deepspeed.checkpoint.autoep_transfer import make_autoep_transfer_request
from deepspeed.checkpoint.autoep_affine import autoep_metadata_to_affine_map
from deepspeed.checkpoint.constants import AUTOEP_PARAM_EP_RANK, AUTOEP_PARAM_LOGICAL_SHAPE, DS_AUTOEP_UC_META
from deepspeed.module_inject.auto_ep_config import AutoEPConfig, MoELayerSpec
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

ROOT = Path(__file__).resolve().parents[3]
PRIMES = (251, 241, 239, 233, 229)
GATE_SOURCES = ('deepspeed/checkpoint/affine_transfer.py', 'deepspeed/checkpoint/affine_transfer_executor.py',
                'deepspeed/checkpoint/autoep_transfer.py', 'tests/distributed/checkpoint/affine_handoff_h100.py',
                'deepspeed/comm/comm.py', 'deepspeed/comm/torch.py')
CONTROL_SEQUENCE = 0
CONTROL_BROKEN = False
JOURNAL = None
MAX_CONTROL_BYTES = 16 << 20
COLLECTIVE_TIMEOUT = 120


def journal(event, **fields):
    if JOURNAL is not None:
        with JOURNAL.open('a') as stream:
            stream.write(json.dumps({'event': event, 'epoch': time.time(), **fields}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def require_passed_gate(directory):
    result = json.loads((directory / 'correctness.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    cases = manifest.get('cases', [])
    if (result.get('status') != 'PASS' or not result.get('eight_gpu_correctness_gate')
            or manifest.get('world_size') != 8 or manifest.get('scope') != 'cross-node' or len(cases) != 8
            or set(cases) != {f'U{index}'
                              for index in range(1, 9)}):
        raise ValueError('all eight prerequisite cases must pass before the consumer or timing run')
    hashes = manifest.get('source_sha256', {})
    if set(hashes) != set(GATE_SOURCES):
        raise ValueError('the correctness result does not cover the required source inventory')
    for name in GATE_SOURCES:
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError('the source differs from the passed correctness source')


def gather(value, phase='gather'):
    """Keep bounded control messages on CPU/Gloo, independent of the CUDA/NCCL schedule."""
    global CONTROL_SEQUENCE, CONTROL_BROKEN
    if CONTROL_BROKEN:
        raise RuntimeError('the control channel failed; no further collective is admitted')
    error, encoded = None, b''
    try:
        encoded = pickle.dumps({'phase': phase, 'value': value}, protocol=4)
        if len(encoded) > MAX_CONTROL_BYTES:
            raise ValueError('control message exceeds the fixed size limit')
    except Exception as caught:
        error = repr(caught)
    journal('CONTROL_ENTER', phase=phase, sequence=CONTROL_SEQUENCE, serialization_error=error)
    try:
        header = torch.tensor([CONTROL_SEQUENCE, len(encoded), int(error is not None)],
                              dtype=torch.int64,
                              device='cpu')
        headers = [torch.empty_like(header) for _ in range(dist.get_world_size())]
        dist.all_gather(headers, header)
        lengths = [int(row[1]) for row in headers]
        if (any(int(row[0]) != CONTROL_SEQUENCE or int(row[2]) for row in headers)
                or any(length <= 0 or length > MAX_CONTROL_BYTES for length in lengths)):
            raise RuntimeError('control sequence, serialization or bounded length agreement failed')
        payload = torch.zeros(max(lengths), dtype=torch.uint8, device='cpu')
        payload[:len(encoded)].copy_(torch.frombuffer(bytearray(encoded), dtype=torch.uint8))
        received = [torch.empty_like(payload) for _ in headers]
        dist.all_gather(received, payload)
        messages = [pickle.loads(bytes(row[:length].tolist())) for row, length in zip(received, lengths)]
        if any(message['phase'] != phase for message in messages):
            raise RuntimeError(f'control phases disagree: {[message["phase"] for message in messages]}')
        CONTROL_SEQUENCE += 1
        return [message['value'] for message in messages]
    except Exception:
        CONTROL_BROKEN = True
        journal('CONTROL_FAILED', phase=phase, traceback=traceback.format_exc())
        raise


def coordinated(phase, function):
    """Agree on rank-local failures before any rank starts the next data collective."""
    deadline = os.environ.get('UC_COMPUTE_DEADLINE')
    ready = deadline is None or time.time() + COLLECTIVE_TIMEOUT < float(deadline)
    if not all(gather(ready, phase=phase + '/admission')):
        raise RuntimeError(f'{phase}: insufficient original compute time for a bounded collective')
    value, error = None, None
    journal('PHASE_START', phase=phase)
    try:
        value = function()
    except Exception:
        error = traceback.format_exc()
    journal('PHASE_FINISH', phase=phase, error=error)
    failures = gather(error, phase=phase)
    if any(failures):
        raise RuntimeError(f'{phase} failed: {failures}')
    return value


def initialize(timeout_seconds):
    global COLLECTIVE_TIMEOUT
    COLLECTIVE_TIMEOUT = timeout_seconds
    if not all(key in os.environ for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')):
        raise RuntimeError('launch one external worker per GPU; this harness does not spawn workers')
    if os.environ.get('TORCH_NCCL_BLOCKING_WAIT') != '1':
        raise RuntimeError('correctness workers require NCCL errors to return through blocking wait')
    local_rank = int(os.environ['LOCAL_RANK'])
    accelerator = get_accelerator()
    if local_rank >= accelerator.device_count():
        raise RuntimeError('LOCAL_RANK does not name a visible accelerator')
    accelerator.set_device(local_rank)
    dist.init_distributed(dist_backend='cpu:gloo,cuda:nccl',
                          init_method='env://',
                          timeout=timedelta(seconds=timeout_seconds))
    properties = torch.cuda.get_device_properties(local_rank)  #ignore-cuda: physical CUDA UUID has no accelerator API
    identity = {
        'rank': dist.get_rank(),
        'local_rank': local_rank,
        'host': socket.gethostname(),
        'uuid': str(properties.uuid),
        'name': properties.name,
        'total_memory': properties.total_memory
    }
    identities = gather(identity)
    if len({item['uuid'] for item in identities}) != dist.get_world_size():
        raise RuntimeError('global ranks do not use distinct physical GPUs')
    if any('H100' not in item['name'] for item in identities):
        raise RuntimeError('this hardware gate requires H100 GPUs')
    return torch.device(accelerator.current_device_name()), identities


def split_sizes(extent, count):
    return [extent // count + (rank < extent % count) for rank in range(count)]


def endpoints_for(case, world_size):
    half = world_size // 2
    if case in ('U1', 'U6', 'U7', 'U8'):
        return TransferEndpoints(dict(enumerate(range(half))), dict(enumerate(range(half, world_size))))
    if case == 'U2':
        return TransferEndpoints(dict(enumerate(range(max(1, half // 2)))), dict(enumerate(range(half, world_size))))
    if case == 'U3':
        return TransferEndpoints(dict(enumerate(range(half))), dict(enumerate(range(world_size))))
    if case == 'U4':
        return TransferEndpoints(dict(enumerate(range(world_size))), dict(enumerate(range(half, world_size))))
    if case == 'U5':
        return TransferEndpoints(dict(enumerate(range(world_size))),
                                 {rank: (rank + half) % world_size
                                  for rank in range(world_size)})
    raise ValueError(f'unknown matrix case {case}')


def maps_for(shape, endpoints, replicated=False, same_layout=False):
    if replicated:
        return replicated_map(shape, len(endpoints.source)), replicated_map(shape, len(endpoints.target))
    source = contiguous_split_map(shape, split_sizes(shape[0], len(endpoints.source)), 0)
    target = source if same_layout else contiguous_split_map(shape, split_sizes(shape[-1], len(endpoints.target)),
                                                             len(shape) - 1)
    return source, target


def marker_moduli(elements):
    covered, result = 1, []
    for prime in PRIMES:
        result.append(prime)
        covered *= prime
        if covered >= elements:
            return result
    raise ValueError('marker identity range exceeds the independent modular oracle')


def tensor_fixture(shape, endpoints, dtype, device, modulus, parameter_count=1, replicated=False, same_layout=False):
    source_map, target_map = maps_for(shape, endpoints, replicated, same_layout)
    requests, targets, expected, guards = [], [], [], []
    for item in range(parameter_count):
        values = torch.arange(math.prod(shape), dtype=torch.int64, device=device)
        full = ((values + item * math.prod(shape)) % modulus).to(dtype).view(shape)
        sources = {
            location: source_map.extract(full, location).contiguous()
            for location, owner in endpoints.source.items() if owner == dist.get_rank()
        }
        local_targets = {}
        for location, owner in endpoints.target.items():
            if owner != dist.get_rank():
                continue
            local_shape = target_map.shard_shapes[location]
            storage = torch.full((math.prod(local_shape) + 16, ), -1, dtype=dtype, device=device)
            local_targets[location] = storage[8:-8].view(local_shape)
            expected.append((local_targets[location], target_map.extract(full, location)))
            guards.append(storage)
        requests.append(
            TransferRequest(f'parameter.{item}', plan_transfer(target_map, source_map), sources, local_targets))
        targets.extend(local_targets.values())
    get_accelerator().synchronize()
    return requests, targets, expected, guards


def tensor_case(case, shape, endpoints, dtype, device, peer_bytes, parameter_count=1, replicated=False):
    trials = []
    for modulus in marker_moduli(math.prod(shape) * parameter_count):
        phase = f'{case}/modulus-{modulus}'
        requests, _, expected, guards = coordinated(
            phase + '/prepare', lambda: tensor_fixture(shape, endpoints, dtype, device, modulus, parameter_count,
                                                       replicated, case == 'U5'))

        def transfer_and_check():
            started = time.perf_counter()
            stats = execute_transfers(requests, endpoints, dtype, StagingBudget(peer_bytes))
            elapsed = time.perf_counter() - started
            if case in ('U1', 'U2', 'U5', 'U7') and stats.local_copy_bytes:
                raise AssertionError('disjoint or owner-swap cases must move every payload across processes')
            errors = sum(int((actual != wanted).sum().item()) for actual, wanted in expected)
            guard_errors = sum(
                int((storage[:8] != -1).sum().item()) + int((storage[-8:] != -1).sum().item()) for storage in guards)
            if errors or guard_errors:
                raise AssertionError(f'{case}: {errors} payload errors and {guard_errors} guard errors')
            return {
                'modulus': modulus,
                'target_ready_seconds': elapsed,
                'stats': asdict(stats),
                'error_elements': errors,
                'guard_errors': guard_errors
            }

        trials.append(coordinated(phase + '/check', transfer_and_check))
    return {'shape': shape, 'parameters': parameter_count, 'peer_bytes': peer_bytes, 'trials': trials}


def rejection_case(fault, endpoints, dtype, device, peer_bytes):
    shape = (16, 17)
    requests, targets, _, guards = coordinated(
        f'U6/{fault}/prepare', lambda: tensor_fixture(shape, endpoints, dtype, device, 251, parameter_count=2))
    source_owner, target_owner = endpoints.source[0], endpoints.target[0]
    budget = StagingBudget(peer_bytes)
    if fault == 'missing_source' and dist.get_rank() == source_owner:
        requests[1].source_buffers.pop(0)
    elif fault == 'missing_target' and dist.get_rank() == target_owner:
        requests[1].target_buffers.pop(0)
    elif fault == 'dtype' and dist.get_rank() == source_owner:
        requests[1].source_buffers[0] = requests[1].source_buffers[0].to(torch.float64)
    elif fault == 'budget' and dist.get_rank() == source_owner:
        budget = StagingBudget(peer_bytes + 2)
    elif fault == 'key_order' and dist.get_rank() == source_owner:
        requests.reverse()
    elif fault == 'malformed_stride' and dist.get_rank() == source_owner:
        plan = requests[1].plan
        bad = replace(plan.segments[0], source_strides=(1, ))
        requests[1] = replace(requests[1], plan=replace(plan, segments=(bad, ) + plan.segments[1:]))
    elif fault == 'target_alias' and dist.get_rank() == target_owner:
        requests[1].target_buffers[0] = requests[0].target_buffers[0]
    elif fault == 'adapter_error' and dist.get_rank() == source_owner:
        requests[1] = replace(requests[1], plan=None, preflight_error='invalid local AutoEP metadata')
    rejected = False
    try:
        execute_transfers(requests, endpoints, dtype, budget)
    except AffineTransferExecutionError:
        rejected = True
    untouched = all(bool(torch.all(target == -1).item()) for target in targets)
    intact = all(
        bool(torch.all(storage[:8] == -1).item()) and bool(torch.all(storage[-8:] == -1).item()) for storage in guards)
    if not rejected or not untouched or not intact:
        raise AssertionError(f'{fault}: collective refusal or sentinel preservation failed')
    return {'fault': fault, 'rejected': rejected, 'targets_untouched': untouched, 'guards_intact': intact}


class TinySourceMoE(nn.Module):

    def __init__(self, experts):
        super().__init__()
        self.gate = nn.Linear(16, experts, bias=False)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.randn(experts, 64, 16) * 0.2)
        self.experts.down_proj = nn.Parameter(torch.randn(experts, 16, 32) * 0.2)


def consumer_spec(experts):
    return MoELayerSpec(moe_module_name='model.layers.0.mlp',
                        model_family='mixtral',
                        router_name='gate',
                        experts_name='experts',
                        expert_storage='fused_3d',
                        expert_w1_name='gate_up_proj',
                        expert_w2_name='down_proj',
                        expert_w3_name=None,
                        num_experts=experts,
                        top_k=2,
                        hidden_size=16,
                        ffn_hidden_size=32,
                        score_func='softmax',
                        score_apply='post',
                        route_norm=True,
                        gate_bias=False,
                        return_router_logits=False,
                        router_logits_capture_target='none',
                        router_logits_capture_index=None,
                        router_logits_capture_layer_name=None,
                        has_shared_experts=False,
                        shared_experts_name='')


def expert_parameters(layer):
    if layer is None:
        return {}
    return {
        name: parameter
        for name, parameter in layer.named_parameters() if getattr(parameter, DS_AUTOEP_UC_META, None) is not None
    }


@torch.no_grad()
def consumer_case(endpoints, dtype, device, peer_bytes):
    source_ranks, target_ranks = list(endpoints.source.values()), list(endpoints.target.values())
    source_group = dist.new_group(ranks=source_ranks, timeout=timedelta(seconds=COLLECTIVE_TIMEOUT))
    target_group = dist.new_group(ranks=target_ranks, timeout=timedelta(seconds=COLLECTIVE_TIMEOUT))
    torch.manual_seed(1729)
    source_module = TinySourceMoE(dist.get_world_size() * 2).to(device=device, dtype=dtype).eval()
    source_module.requires_grad_(False)
    spec = consumer_spec(dist.get_world_size() * 2)
    config = AutoEPConfig(enabled=True,
                          use_grouped_mm=False,
                          disable_triton_grouped_mm=True,
                          comm_backend='comm',
                          row_weighting_impl='eager',
                          combine_impl='weighted_sum')
    source_layer, target_layer = None, None
    if dist.get_rank() in source_ranks:
        local = source_ranks.index(dist.get_rank())
        source_layer = AutoEPMoELayer(spec, copy.deepcopy(source_module), len(source_ranks), local, config).eval()
        source_layer.ep_group = source_group
    if dist.get_rank() in target_ranks:
        local = target_ranks.index(dist.get_rank())
        target_layer = AutoEPMoELayer(spec, copy.deepcopy(source_module), len(target_ranks), local, config).eval()
        target_layer.ep_group = target_group
    sources, targets = expert_parameters(source_layer), expert_parameters(target_layer)
    local_metadata = {
        'source': {
            key: getattr(parameter, DS_AUTOEP_UC_META)
            for key, parameter in sources.items()
        },
        'target': {
            key: getattr(parameter, DS_AUTOEP_UC_META)
            for key, parameter in targets.items()
        }
    }
    metadata = gather(local_metadata, phase='consumer/metadata')
    source_metadata = metadata[source_ranks[0]]['source']
    target_metadata = metadata[target_ranks[0]]['target']
    identities = {key: (id(parameter), parameter.data_ptr()) for key, parameter in targets.items()}
    reference = None
    if target_layer is not None:
        reference = AutoEPMoELayer(spec, copy.deepcopy(source_module), 1, 0, config).eval()
    inputs = torch.arange(16 * 8, dtype=torch.float32, device=device).view(2, 4, 16) / 128
    inputs = inputs.to(dtype)
    reports = []
    for generation in range(3):

        def prepare_generation():
            if generation:
                for parameter in sources.values():
                    parameter.mul_(0.5)
                if reference is not None:
                    for parameter in expert_parameters(reference).values():
                        parameter.mul_(0.5)
            for parameter in targets.values():
                parameter.fill_(float('nan'))
            requests = []
            for key in sorted(source_metadata):
                local_sources = {source_ranks.index(dist.get_rank()): sources[key]} if sources else {}
                local_targets = {target_ranks.index(dist.get_rank()): targets[key]} if targets else {}
                requests.append(
                    make_autoep_transfer_request(key, source_metadata[key], target_metadata[key], local_sources,
                                                 local_targets))
            return requests

        requests = coordinated(f'U8/generation-{generation}/prepare', prepare_generation)
        get_accelerator().synchronize()
        caller = get_accelerator().Stream()

        def transfer_and_consume():
            started = time.perf_counter()
            with get_accelerator().stream(caller):
                stats = execute_transfers(requests, endpoints, dtype, StagingBudget(peer_bytes))
            ready_seconds = time.perf_counter() - started
            report = {'generation': generation, 'target_ready_seconds': ready_seconds, 'stats': asdict(stats)}
            if target_layer is not None:
                # No global barrier or completion event is inserted before this caller-stream consumer.
                actual = target_layer(inputs)
                get_accelerator().synchronize()
                report['first_forward_seconds'] = time.perf_counter() - started
                wanted = reference(inputs)
                rtol, atol, l2_limit = (1e-2, 1e-4, 1e-2) if dtype == torch.bfloat16 else (1e-5, 1e-6, 1e-5)
                torch.testing.assert_close(actual, wanted, rtol=rtol, atol=atol)
                report['tolerances'] = {'rtol': rtol, 'atol': atol, 'relative_l2_limit': l2_limit}
                difference = actual.float() - wanted.float()
                report['max_abs_error'] = float(difference.abs().max().item())
                report['relative_l2'] = float(difference.norm().item() / max(wanted.float().norm().item(), 1e-12))
                if report['relative_l2'] > l2_limit:
                    raise AssertionError('consumer relative L2 exceeds the frozen numerical limit')
                reference_parameters = expert_parameters(reference)
                for key, parameter in targets.items():
                    metadata_ = getattr(parameter, DS_AUTOEP_UC_META)
                    map_ = autoep_metadata_to_affine_map(metadata_, metadata_[AUTOEP_PARAM_LOGICAL_SHAPE], key)
                    expected_weight = map_.extract(reference_parameters[key], metadata_[AUTOEP_PARAM_EP_RANK])
                    if not torch.equal(parameter, expected_weight):
                        raise AssertionError('the target expert weights differ from the frozen reference')
                    if (id(parameter), parameter.data_ptr()) != identities[key]:
                        raise AssertionError('target Parameter identity or storage changed')
                    if bool(torch.isnan(parameter).any().item()):
                        raise AssertionError('the consumer read an unwritten expert parameter')
            return report

        report = coordinated(f'U8/generation-{generation}/consumer', transfer_and_consume)
        reports.append(report)
    return {
        'consumer': 'AutoEPMoELayer with runtime-produced uniform placement',
        'generations': reports,
        'full_model': 'NOT_RUN',
        'gradient_or_training_resume': 'NOT_RUN'
    }


def run_case(name, function, records):
    started = time.perf_counter()
    try:
        get_accelerator().reset_peak_memory_stats()
        detail = function()
        record = {'case': name, 'rank': dist.get_rank(), 'status': 'PASS', 'detail': detail}
    except Exception as error:
        record = {
            'case': name,
            'rank': dist.get_rank(),
            'status': 'FAIL',
            'error': repr(error),
            'traceback': traceback.format_exc()
        }
    record['wall_seconds'] = time.perf_counter() - started
    try:
        record['peak_allocated_bytes'] = get_accelerator().max_memory_allocated()
        record['peak_reserved_bytes'] = get_accelerator().max_memory_reserved()
    except Exception:
        record['status'] = 'FAIL'
        record['memory_metrics_error'] = traceback.format_exc()
    record[
        'memory_scope'] = 'case PyTorch peak with process-resident allocator cache; excludes non-PyTorch allocations'
    journal('CASE_RESULT', record=record)
    records.append(record)
    rows = gather(record, phase=f'case/{name}')
    records[-1:] = rows
    if any(row['status'] != 'PASS' for row in rows):
        raise AssertionError(f'{name} failed; rank records were preserved')


def main():
    global JOURNAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=('prerequisite', 'cross-node'), default='cross-node')
    parser.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--peer-mib', type=int, default=8)
    parser.add_argument('--timeout-seconds', type=int, default=120)
    parser.add_argument('--cases',
                        nargs='+',
                        choices=[f'U{index}' for index in range(1, 9)],
                        default=[f'U{index}' for index in range(1, 9)])
    args = parser.parse_args()
    if len(args.cases) != len(set(args.cases)):
        raise ValueError('correctness cases may not be repeated in place of omitted cases')
    args.output.mkdir(parents=True, exist_ok=True)
    JOURNAL = args.output / f'rank-{int(os.environ["RANK"]):02d}.jsonl'
    JOURNAL.open('x').close()
    journal('ENTRY', dtype=args.dtype, cases=args.cases)
    device, identities = initialize(args.timeout_seconds)
    world_size = dist.get_world_size()
    if world_size not in (2, 4, 8):
        raise RuntimeError('this initial harness admits 2/4 GPU prerequisites and the 8 GPU gate')
    source_hosts = {item['host'] for item in identities if item['rank'] < world_size // 2}
    target_hosts = {item['host'] for item in identities if item['rank'] >= world_size // 2}
    cross_node = (world_size == 8 and len(source_hosts) == len(target_hosts) == 1
                  and source_hosts.isdisjoint(target_hosts))
    if args.scope == 'cross-node' and not cross_node:
        raise RuntimeError('cross-node gate requires eight distinct GPUs on two nodes')
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    peer_bytes = args.peer_mib << 20
    records, status = [], 'FAIL'
    try:
        for case in args.cases:
            endpoints = endpoints_for(case, world_size)
            if case == 'U6':
                for fault in ('missing_source', 'missing_target', 'dtype', 'budget', 'key_order', 'malformed_stride',
                              'target_alias', 'adapter_error'):
                    run_case(f'U6/{fault}',
                             lambda fault=fault: rejection_case(fault, endpoints, dtype, device, peer_bytes),
                             records)
            elif case == 'U7':
                for shape in ((8, 17, 29), (16, 17, 1), (8, 5, 7, 3)):
                    run_case(f'U7/{shape}',
                             lambda shape=shape: tensor_case(case, shape, endpoints, dtype, device, 128),
                             records)
                run_case(
                    'U7/last-axis-counterexample',
                    lambda: tensor_case(case, (4, 4096, 4096), endpoints, dtype, device, peer_bytes, replicated=True),
                    records)
            elif case == 'U8':
                run_case('U8/disjoint', lambda: consumer_case(endpoints, dtype, device, peer_bytes), records)
                resized = endpoints_for('U3', world_size)
                run_case('U8/resize', lambda: consumer_case(resized, dtype, device, peer_bytes), records)
            else:
                shape = (16, 17)
                run_case(case,
                         lambda: tensor_case(case, shape, endpoints, dtype, device, peer_bytes, parameter_count=2),
                         records)
        status = 'PASS'
    finally:
        if dist.get_rank() == 0:
            manifest = {
                'scope': args.scope,
                'world_size': world_size,
                'physical_devices': identities,
                'torch': torch.__version__,
                'cuda': torch.version.cuda,
                'nccl': torch.cuda.nccl.version(),  #ignore-cuda: NCCL version has no accelerator API
                'source_sha256': {
                    path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                    for path in GATE_SOURCES
                },
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'dtype': args.dtype,
                'control_backend': 'cpu:gloo,cuda:nccl',
                'nccl_blocking_wait': os.environ['TORCH_NCCL_BLOCKING_WAIT'],
                'peer_bytes': peer_bytes,
                'cases': args.cases
            }
            (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            result = {
                'status': status,
                'scope': args.scope,
                'records': records,
                'eight_gpu_correctness_gate': status == 'PASS' and cross_node and len(args.cases) == 8
                and set(args.cases) == {f'U{index}'
                                        for index in range(1, 9)},
                'full_model': 'NOT_RUN',
                'performance_gate': 'NOT_RUN',
                'expansion_admitted': False
            }
            (args.output / 'correctness.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({key: value for key, value in result.items() if key != 'records'}), flush=True)
        journal('FINAL_RESULT', status=status, records=records)
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        journal('ENTRY_FAILED', traceback=traceback.format_exc())
        raise

# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Full-checkpoint expert handoff followed by concurrent cached generation.

Requests run as finite cohorts. This measures an in-process consumer, including
prefill and decode, and does not stand in for an HTTP server or a training restart.
"""

import argparse
import copy
import hashlib
import importlib.util
from importlib.metadata import version
import inspect
import json
import math
import random
import resource
import statistics
import time
import zlib
from pathlib import Path, PurePosixPath

import torch
from packaging.version import Version

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine_transfer_executor import StagingBudget, TransferEndpoints, execute_transfers, plan_digest
from deepspeed.checkpoint.autoep_transfer import make_autoep_transfer_request
from deepspeed.checkpoint.constants import AUTOEP_PARAM_LOGICAL_SHAPE, DS_AUTOEP_UC_META
from deepspeed.module_inject.auto_ep import AutoEP
from deepspeed.module_inject.auto_ep_config import parse_autoep_config

ROOT = Path(__file__).resolve().parents[3]
CONCURRENCIES = (32, 64, 128)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(snapshot, manifest_path, manifest_sha, requests_path, requests_sha):
    if file_sha256(manifest_path) != manifest_sha or file_sha256(requests_path) != requests_sha:
        raise ValueError('the model manifest or request trace differs from the frozen input')
    manifest = json.loads(manifest_path.read_text())
    revision = manifest.get('revision', '')
    if len(revision) != 40 or any(character not in '0123456789abcdef' for character in revision):
        raise ValueError('a complete immutable model revision is required')
    files = manifest.get('files', {})
    if 'config.json' not in files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('the manifest must cover the full checkpoint and configuration')
    if manifest.get('parameter_count', 0) < 10_000_000_000 or manifest.get('moe_layers', 0) < 20:
        raise ValueError('a reduced model fixture cannot satisfy the full-checkpoint gate')
    required = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob('*')
        if path.is_file() and path.suffix in ('.json', '.safetensors', '.model', '.txt', '.py', '.jinja')
    }
    if not required.issubset(files):
        raise ValueError('checkpoint, tokenizer, or local model-code files are absent from the manifest')
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('model manifest entries must be relative file names')
        if file_sha256(snapshot / name) != expected:
            raise ValueError('a checkpoint asset differs from the frozen SHA256')
    requests = [json.loads(line) for line in requests_path.read_text().splitlines() if line.strip()]
    if len(requests) < max(CONCURRENCIES) or len({row['id'] for row in requests}) != len(requests):
        raise ValueError('at least 128 independently identified requests are required')
    for row in requests:
        if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError('every request must contain a nonempty natural-language prompt')
    return manifest, requests


def load_harness():
    path = ROOT / 'tests/distributed/checkpoint/affine_handoff_h100.py'
    spec = importlib.util.spec_from_file_location('affine_handoff_prerequisite', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_route(path, expected_sha, requests, endpoints, budget):
    path_ = ROOT / 'benchmarks/checkpoint/affine_handoff_h100.py'
    spec = importlib.util.spec_from_file_location('affine_handoff_baseline_loader', path_)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    module, metadata = loader.load_legacy(path, expected_sha)

    def route():
        module.execute_transfers(requests, endpoints, torch.bfloat16, module.StagingBudget(budget.bytes_per_peer))
        get_accelerator().current_stream().synchronize()

    return route, metadata


def parameter_digest(parameters):
    digest = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        digest.update(json.dumps((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        flat = tensor.detach().contiguous().view(-1)
        chunk = max(1, (16 << 20) // tensor.element_size())
        for start in range(0, flat.numel(), chunk):
            # Byte views preserve BF16 bits; conversion to float would weaken the exact oracle.
            digest.update(flat[start:start + chunk].view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def model_parameters(model):
    experts, frozen = {}, {}
    for name, parameter in model.named_parameters():
        destination = experts if getattr(parameter, DS_AUTOEP_UC_META, None) is not None else frozen
        destination[name] = parameter
    state = model.state_dict()
    for name, buffer in model.named_buffers():
        if name in state:
            frozen[name] = buffer
    return experts, frozen


def move_model_preserving_expert_metadata(model, device):
    metadata = {
        name: copy.deepcopy(getattr(parameter, DS_AUTOEP_UC_META))
        for name, parameter in model.named_parameters() if getattr(parameter, DS_AUTOEP_UC_META, None) is not None
    }
    # Device conversion can replace Parameters and discard their Python attributes.
    model.to(device)
    parameters = dict(model.named_parameters())
    for name, attached in metadata.items():
        setattr(parameters[name], DS_AUTOEP_UC_META, attached)
    return model


def load_model(args, local_rank, group):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model, loading = AutoModelForCausalLM.from_pretrained(args.snapshot,
                                                          local_files_only=True,
                                                          dtype=torch.bfloat16,
                                                          use_safetensors=True,
                                                          trust_remote_code=args.trust_local_model_code,
                                                          attn_implementation='sdpa',
                                                          output_loading_info=True)
    if any(loading.get(name) for name in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('the complete model checkpoint did not load without missing or mismatched weights')
    if getattr(model.config, 'quantization_config', None) is not None:
        raise ValueError('this expert-update contract requires an unquantized BF16 checkpoint')
    original_parameters = sum(parameter.numel() for parameter in model.parameters())
    if original_parameters != args.model_manifest['parameter_count']:
        raise ValueError('the loaded model is not the complete frozen parameter inventory')
    config = parse_autoep_config({
        'enabled': True,
        'autoep_size': args.ep_size,
        'preset_model': args.preset,
        'comm_backend': 'comm',
        'use_grouped_mm': True,
        'row_weighting_impl': 'eager',
        'combine_impl': 'weighted_sum'
    })
    conversion = AutoEP(model, config)
    specs = conversion.ep_parser()
    if len(specs) != args.model_manifest['moe_layers'] or not specs:
        raise ValueError('AutoEP did not detect the complete frozen MoE layer inventory')
    conversion.replace_moe_layers(specs, args.ep_size, local_rank)
    model.requires_grad_(False)
    model.eval()
    move_model_preserving_expert_metadata(model, get_accelerator().current_device_name())
    for module in model.modules():
        if getattr(module, '_is_autoep_layer', False):
            module.ep_group = group
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot,
                                              local_files_only=True,
                                              trust_remote_code=args.trust_local_model_code)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('the frozen tokenizer supplies neither a pad token nor an EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    signature = inspect.signature(model.forward).parameters
    keep_logits = {}
    if 'logits_to_keep' in signature:
        keep_logits['logits_to_keep'] = 1
    elif 'num_logits_to_keep' in signature:
        keep_logits['num_logits_to_keep'] = 1
    return model, tokenizer, keep_logits, original_parameters


class RoutingCapture:

    def __init__(self, model):
        self.rows = []
        self.handles = [
            module.router.register_forward_hook(self.capture) for module in model.modules()
            if getattr(module, '_is_autoep_layer', False)
        ]

    def capture(self, module, inputs, output):
        self.rows.append(output[1].detach().reshape(-1).clone())

    def take(self):
        if not self.rows:
            raise AssertionError('the full-model forward did not execute any expert routing')
        value = torch.cat(self.rows)
        self.rows.clear()
        return value

    def close(self):
        for handle in self.handles:
            handle.remove()


def exchange_reference(logits, routing, peer, source, device):
    if source:
        header = torch.tensor([*logits.shape, routing.numel()], dtype=torch.int64, device=device)
    else:
        header = torch.empty(3, dtype=torch.int64, device=device)
    function = dist.isend if source else dist.irecv
    for work in dist.batch_isend_irecv([dist.P2POp(function, header, peer)]):
        work.wait()
    if not source:
        batch, vocabulary, routes = header.tolist()
        logits = torch.empty((batch, vocabulary), dtype=torch.float32, device=device)
        routing = torch.empty(routes, dtype=torch.int64, device=device)
    operations = [dist.P2POp(function, tensor, peer) for tensor in (logits, routing)]
    for work in dist.batch_isend_irecv(operations):
        work.wait()
    return logits, routing


def next_inputs(inputs, tokens):
    mask = inputs['attention_mask']
    return {
        'input_ids': tokens.reshape(-1, 1),
        'attention_mask': torch.cat((mask, mask.new_ones((mask.shape[0], 1))), dim=1)
    }


def position_ids(inputs):
    mask = inputs['attention_mask']
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return positions[:, -inputs['input_ids'].shape[1]:]


@torch.no_grad()
def make_reference(model, inputs, keep_logits, source, peer, device, decode_tokens):
    capture = RoutingCapture(model) if source else None
    cache, trace, current = None, [], inputs
    try:
        for step in range(decode_tokens):
            logits, routing = None, None
            if source:
                outputs = model(**current,
                                **keep_logits,
                                position_ids=position_ids(current),
                                past_key_values=cache,
                                use_cache=True)
                cache = outputs.past_key_values
                if cache is None:
                    raise AssertionError('the full-model decode did not return its KV cache')
                logits, routing = outputs.logits[:, -1].float().contiguous(), capture.take()
            logits, routing = exchange_reference(logits, routing, peer, source, device)
            tokens = logits.argmax(-1)
            trace.append({
                'input_ids': current['input_ids'].cpu(),
                'attention_mask': current['attention_mask'].cpu(),
                'logits': logits.cpu(),
                'routing': routing.cpu(),
                'next_tokens': tokens.cpu()
            })
            current = next_inputs(current, tokens)
        get_accelerator().synchronize()
    finally:
        if capture is not None:
            capture.close()
    return trace


@torch.no_grad()
def consume(model, trace, keep_logits, device, validate):
    capture = RoutingCapture(model) if validate else None
    cache, outputs, step_seconds, generated = None, [], [], []
    started, previous = time.perf_counter(), 0.0
    inputs = {name: trace[0][name].to(device) for name in ('input_ids', 'attention_mask')}
    try:
        for step in range(len(trace)):
            result = model(**inputs,
                           **keep_logits,
                           position_ids=position_ids(inputs),
                           past_key_values=cache,
                           use_cache=True)
            cache = result.past_key_values
            if cache is None:
                raise AssertionError('the target decode did not retain its KV cache')
            logits = result.logits[:, -1].float().contiguous()
            tokens = logits.argmax(-1)
            generated.append(tokens)
            routing = capture.take() if capture is not None else None
            if device.type != 'cpu':
                get_accelerator().current_stream().synchronize()
            completed = time.perf_counter() - started
            step_seconds.append(completed - previous)
            previous = completed
            if validate:
                outputs.append((logits, routing))
            if step + 1 < len(trace):
                inputs = next_inputs(inputs, tokens)
        if device.type != 'cpu':
            get_accelerator().current_stream().synchronize()
    finally:
        if capture is not None:
            capture.close()
    return outputs, step_seconds, torch.stack(generated)


def validate_outputs(outputs, trace, device):
    max_error, squared_error, squared_reference, wrong_routes, wrong_tokens = 0.0, 0.0, 0.0, 0, 0
    for (actual, routing), row in zip(outputs, trace):
        wanted = row['logits'].to(device)
        torch.testing.assert_close(actual, wanted, rtol=1e-2, atol=1e-4)
        difference = actual - wanted
        max_error = max(max_error, difference.abs().max().item())
        squared_error += difference.square().sum().item()
        squared_reference += wanted.square().sum().item()
        wrong_routes += (routing != row['routing'].to(device)).sum().item()
        wrong_tokens += (actual.argmax(-1) != row['next_tokens'].to(device)).sum().item()
    relative_l2 = math.sqrt(squared_error / max(squared_reference, 1e-24))
    if len(outputs) != len(trace) or relative_l2 > 1e-2 or wrong_routes or wrong_tokens:
        raise AssertionError('full-model logits, routing, or greedy generation differs from the frozen source')
    return {
        'max_abs_error': max_error,
        'relative_l2': relative_l2,
        'wrong_routes': wrong_routes,
        'wrong_tokens': wrong_tokens,
        'rtol': 1e-2,
        'atol': 1e-4,
        'relative_l2_limit': 1e-2
    }


class RebuildReference:
    """Coalesce a bounded group of complete parameters, gather, rebuild, and extract."""

    def __init__(self, requests, metadata, endpoints, source_group, dtype, device, coalesce_bytes, budget):
        self.requests, self.endpoints, self.source_group = requests, endpoints, source_group
        self.source = dist.get_rank() in endpoints.source.values()
        self.location = dist.get_rank() % len(endpoints.source)
        self.ep_size, self.dtype, self.device = len(endpoints.source), dtype, device
        self.groups, group, size = [], [], 0
        for request in requests:
            shape = tuple(metadata[request.key][AUTOEP_PARAM_LOGICAL_SHAPE])
            local_shape = (shape[0] // self.ep_size, *shape[1:])
            if shape[0] % self.ep_size:
                raise ValueError('the coalesced baseline requires this runtime uniform expert placement')
            count = math.prod(local_shape)
            if group and (size + count) * self.ep_size * 2 > coalesce_bytes:
                self.groups.append(group)
                group, size = [], 0
            group.append((request, count, size))
            size += count
        if group:
            self.groups.append(group)
        capacity = max(sum(count for _, count, _ in group) for group in self.groups)
        self.full = torch.empty(capacity * self.ep_size, dtype=dtype, device=device)
        self.packed = torch.empty(capacity, dtype=dtype, device=device) if self.source else None
        self.gathered = [torch.empty_like(self.packed) for _ in range(self.ep_size)] if self.source else []
        frame = (str(dtype),
                 tuple((request.key, plan_digest(request.plan, endpoints, budget, 2)) for request in requests))
        self.digest = zlib.crc32(repr(frame).encode())

    def __call__(self):
        rejected = any(not buffer.is_contiguous() or buffer.dtype != self.dtype or buffer.device != self.device
                       for request in self.requests
                       for buffer in list(request.source_buffers.values()) + list(request.target_buffers.values()))
        local = torch.tensor([self.digest, int(rejected)], dtype=torch.int64, device=self.device)
        seen = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(seen, local)
        if any(row[1].item() for row in seen) or len({row[0].item() for row in seen}) != 1:
            raise ValueError('the rebuild reference rejected the frozen handoff contract')
        for group in self.groups:
            local_count = sum(count for _, count, _ in group)
            full = self.full[:local_count * self.ep_size]
            if self.source:
                for request, count, offset in group:
                    self.packed[offset:offset + count].copy_(request.source_buffers[self.location].view(-1))
                gathered = [buffer[:local_count] for buffer in self.gathered]
                dist.all_gather(gathered, self.packed[:local_count], group=self.source_group)
                if dist.get_rank() == self.endpoints.source[0]:
                    for _, count, offset in group:
                        for location, shard in enumerate(gathered):
                            start = offset * self.ep_size + location * count
                            full[start:start + count].copy_(shard[offset:offset + count])
            dist.broadcast(full, src=self.endpoints.source[0])
            if not self.source:
                for request, count, offset in group:
                    start = offset * self.ep_size + self.location * count
                    request.target_buffers[self.location].view(-1).copy_(full[start:start + count])
        get_accelerator().current_stream().synchronize()


def memory_state():
    accelerator = get_accelerator()
    free, total = torch.cuda.mem_get_info()  #ignore-cuda: whole-device free memory has no accelerator API
    return {
        'allocated_bytes': accelerator.memory_allocated(),
        'reserved_bytes': accelerator.memory_reserved(),
        'peak_allocated_bytes': accelerator.max_memory_allocated(),
        'peak_reserved_bytes': accelerator.max_memory_reserved(),
        'whole_device_free_bytes': free,
        'whole_device_total_bytes': total,
        'cpu_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    }


def percentile(values, quantile):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def run_cycle(route, model, trace, keep_logits, source, device, validate):
    caller = get_accelerator().Stream()
    started = time.perf_counter()
    with get_accelerator().stream(caller):
        route()
    ready = time.perf_counter() - started
    outputs, steps, tokens = ([], [], None) if source else consume(model, trace, keep_logits, device, validate)
    # A repeated update may start only after every target finished its consumer cohort.
    completed = torch.tensor(int(not source), dtype=torch.int32, device=device)
    dist.all_reduce(completed)
    if completed.item() != 4:
        raise AssertionError('the source did not receive completion from all four target owners')
    elapsed = time.perf_counter() - started
    return outputs, tokens, {
        'target_ready_seconds': ready,
        'cycle_seconds': elapsed,
        'first_token_seconds': ready + steps[0] if steps else None,
        'step_seconds': steps,
        'source_endpoint': source
    }


def verify_tokens(tokens, trace, source, device):
    if not source:
        expected = torch.stack([row['next_tokens'] for row in trace]).to(device)
        if not torch.equal(tokens, expected):
            raise AssertionError('a measured generation cohort produced different greedy tokens')


def paired_windows(args, harness, routes, model, trace, keep_logits, source, device, concurrency):
    records, names = [], list(routes)
    for route in routes.values():
        for _ in range(8):
            _, tokens, _ = run_cycle(route, model, trace, keep_logits, source, device, False)
            verify_tokens(tokens, trace, source, device)
            del tokens
    warm = harness.gather(memory_state())
    if any(row['whole_device_free_bytes'] < 0.15 * row['whole_device_total_bytes'] for row in warm):
        raise RuntimeError('the warmed full-model workload leaves less than 15 percent whole-device headroom')
    for window in range(args.windows):
        pilots = []
        for name in names:
            _, tokens, sample = run_cycle(routes[name], model, trace, keep_logits, source, device, False)
            verify_tokens(tokens, trace, source, device)
            del tokens
            pilots.append(max(row['cycle_seconds'] for row in harness.gather(sample)))
        # A common frozen count preserves pairing; short windows fail rather than silently count as long ones.
        iterations = max(8, math.ceil(args.window_seconds * 1.10 / min(pilots)))
        for name in names if window % 2 == 0 else reversed(names):
            get_accelerator().reset_peak_memory_stats()
            samples, generated = [], []
            started = time.perf_counter()
            for _ in range(iterations):
                _, tokens, sample = run_cycle(routes[name], model, trace, keep_logits, source, device, False)
                samples.append(sample)
                if not source:
                    generated.append(tokens)
            elapsed = time.perf_counter() - started
            for tokens in generated:
                verify_tokens(tokens, trace, source, device)
            del generated, tokens
            memory = memory_state()
            if memory['allocated_bytes'] > warm[dist.get_rank()]['allocated_bytes'] + (16 << 20):
                raise AssertionError('repeated complete consumer cohorts retain growing live GPU allocations')
            rank_samples = harness.gather({'samples': samples, 'wall_seconds': elapsed, 'memory': memory})
            target = rank_samples[4:]
            if max(row['wall_seconds'] for row in target) < args.window_seconds:
                raise AssertionError('a paired full-model timing window is shorter than its frozen minimum')
            cycles = [
                max(row['samples'][index]['cycle_seconds'] for row in rank_samples) for index in range(iterations)
            ]
            ready = [
                max(row['samples'][index]['target_ready_seconds'] for row in rank_samples)
                for index in range(iterations)
            ]
            first = [
                max(row['samples'][index]['first_token_seconds'] for row in target) for index in range(iterations)
            ]
            # Each admitted cohort has an individual completion record for every active request.
            request_seconds = [
                sum(sample['step_seconds']) for row in target for sample in row['samples']
                for _ in range(concurrency // 4)
            ]
            first_token_seconds = [
                sample['step_seconds'][0] for row in target for sample in row['samples']
                for _ in range(concurrency // 4)
            ]
            inter_token_seconds = [
                duration for row in target for sample in row['samples'] for duration in sample['step_seconds'][1:]
                for _ in range(concurrency // 4)
            ]
            record = {
                'concurrency': concurrency,
                'window': window,
                'route': name,
                'iterations': iterations,
                'rank_samples': rank_samples,
                'max_rank_cycle_seconds': cycles,
                'max_rank_target_ready_seconds': ready,
                'max_rank_first_token_seconds': first,
                'cycle_p50_seconds': statistics.median(cycles),
                'cycle_p95_seconds': percentile(cycles, 0.95),
                'ready_p50_seconds': statistics.median(ready),
                'ready_p95_seconds': percentile(ready, 0.95),
                'requests_per_second': concurrency * iterations / max(row['wall_seconds'] for row in target),
                'tokens_per_second':
                concurrency * len(trace) * iterations / max(row['wall_seconds'] for row in target),
                'request_latency_p95_seconds': percentile(request_seconds, 0.95),
                'ttft_p95_seconds': percentile(first_token_seconds, 0.95),
                'inter_token_p95_seconds': percentile(inter_token_seconds, 0.95),
                'scheduler':
                'finite cohorts; target greedy feedback checked against the frozen source; no network frontend',
                'memory_scope': 'combined resident model and coalesced reference arenas; external telemetry separate'
            }
            records.append(record)
            if dist.get_rank() == 0:
                with (args.output / 'full_model_timings.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
    return records


def paired_summary(records):
    result = []
    generator = random.Random(1729)
    for concurrency in CONCURRENCIES:
        grouped = {}
        for row in records:
            if row['concurrency'] == concurrency:
                grouped.setdefault(row['window'], {})[row['route']] = row
        if not grouped:
            continue
        if any(not {'rebuild_extract', 'candidate'}.issubset(pair) for pair in grouped.values()):
            raise AssertionError('a full-model performance window is missing a paired route')
        gains = [
            pair['candidate']['requests_per_second'] / pair['rebuild_extract']['requests_per_second'] - 1
            for pair in grouped.values()
        ]
        resampled = [statistics.mean(generator.choices(gains, k=len(gains))) for _ in range(5000)]
        result.append({
            'concurrency': concurrency,
            'paired_windows': len(gains),
            'throughput_gain_mean': statistics.mean(gains),
            'throughput_gain_ci95': [percentile(resampled, 0.025),
                                     percentile(resampled, 0.975)],
            'confidence_unit': 'paired window; individual tokens are not independent replicates'
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--model-manifest-sha256', required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--requests-sha256', required=True)
    parser.add_argument('--preset', choices=('deepseek_v3', 'qwen3_moe'), required=True)
    parser.add_argument('--correctness-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=('correctness', 'paired-performance'), default='correctness')
    parser.add_argument('--decode-tokens', type=int, default=16)
    parser.add_argument('--max-prompt-tokens', type=int, default=256)
    parser.add_argument('--peer-mib', type=int, default=8)
    parser.add_argument('--coalesce-mib', type=int, default=1024)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--windows', type=int, default=5)
    parser.add_argument('--window-seconds', type=int, default=60)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--trust-local-model-code', action='store_true')
    parser.add_argument('--legacy-executor', type=Path)
    parser.add_argument('--legacy-sha256')
    args = parser.parse_args()
    if Version(version('transformers')) < Version('5.0.0') or Version(torch.__version__) < Version('2.10.0'):
        raise ValueError('the full-model driver requires Transformers >= 5.0 and PyTorch >= 2.10')
    if (args.generations < 3 or args.decode_tokens < 16 or args.max_prompt_tokens < 32 or args.peer_mib < 1
            or args.coalesce_mib < 1 or args.windows < 5 or args.window_seconds < 60):
        raise ValueError('the frozen acceptance minimums may not be reduced')
    if bool(args.legacy_executor) != bool(args.legacy_sha256):
        raise ValueError('the legacy executor path and immutable SHA256 must be supplied together')
    if args.scope == 'paired-performance' and args.legacy_executor is None:
        raise ValueError('paired performance must probe the published B2 baseline as well')
    args.model_manifest, request_rows = verify_assets(args.snapshot, args.model_manifest, args.model_manifest_sha256,
                                                      args.requests, args.requests_sha256)
    if args.model_manifest.get('preset') != args.preset:
        raise ValueError('the AutoEP preset differs from the frozen model manifest')
    harness = load_harness()
    harness.require_passed_gate(args.correctness_dir)
    device, identities = harness.initialize(args.timeout_seconds)
    if dist.get_world_size() != 8:
        raise ValueError('this first full-model gate requires eight distinct H100 GPUs')
    args.ep_size = 4
    source_ranks, target_ranks = list(range(4)), list(range(4, 8))
    source_hosts = {row['host'] for row in identities if row['rank'] in source_ranks}
    target_hosts = {row['host'] for row in identities if row['rank'] in target_ranks}
    if len(source_hosts) != 1 or len(target_hosts) != 1 or not source_hosts.isdisjoint(target_hosts):
        raise ValueError('the frozen source and consumer must occupy disjoint endpoints')
    endpoints = TransferEndpoints(dict(enumerate(source_ranks)), dict(enumerate(target_ranks)))
    source_group = dist.new_group(source_ranks)
    target_group = dist.new_group(target_ranks)
    source, location = dist.get_rank() < 4, dist.get_rank() % 4
    started = time.perf_counter()
    model, tokenizer, keep_logits, original_parameters = load_model(args, location,
                                                                    source_group if source else target_group)
    load_seconds = time.perf_counter() - started
    experts, frozen = model_parameters(model)
    if not experts or any(parameter.dtype != torch.bfloat16 for parameter in experts.values()):
        raise ValueError('the complete model did not expose normalized BF16 AutoEP expert parameters')
    frozen_digest = parameter_digest(frozen)
    if len(set(harness.gather(frozen_digest))) != 1:
        raise AssertionError('the endpoints did not load identical frozen non-expert weights')
    metadata = {key: getattr(parameter, DS_AUTOEP_UC_META) for key, parameter in experts.items()}
    inventories = harness.gather(metadata)
    source_metadata, target_metadata = inventories[0], inventories[4]
    if set(source_metadata) != set(target_metadata):
        raise ValueError('the complete source and target expert inventories differ')
    requests = [
        make_autoep_transfer_request(key, source_metadata[key], target_metadata[key],
                                     {location: experts[key]} if source else {},
                                     {} if source else {location: experts[key]}) for key in sorted(source_metadata)
    ]
    if any(request.preflight_error for request in requests):
        raise ValueError('the full checkpoint is not normalized for this expert-update contract')
    identities_before = {key: (id(parameter), parameter.data_ptr()) for key, parameter in experts.items()}
    budget = StagingBudget(args.peer_mib << 20)
    baseline = RebuildReference(requests, source_metadata, endpoints, source_group, torch.bfloat16, device,
                                args.coalesce_mib << 20, budget)
    routes = {
        'rebuild_extract': baseline,
        'candidate': lambda: execute_transfers(requests, endpoints, torch.bfloat16, budget)
    }
    legacy, unsupported = None, {}
    if args.legacy_executor is not None:
        old_route, legacy = legacy_route(args.legacy_executor, args.legacy_sha256, requests, endpoints, budget)
        before = parameter_digest(experts) if not source else None
        rejection = None
        try:
            old_route()
        except ValueError as error:
            rejection = str(error)
        rejections = harness.gather(rejection)
        if any(message is not None for message in rejections):
            if any(message is None for message in rejections):
                raise AssertionError('the published B2 baseline rejected on only some ranks')
            if not source and parameter_digest(experts) != before:
                raise AssertionError('a rejected published B2 baseline partially changed the target')
            unsupported['published_b2_transport_port'] = {'status': 'UNSUPPORTED_OR_REJECTED', 'ranks': rejections}
        else:
            routes['published_b2_transport_port'] = old_route
    else:
        unsupported['published_b2'] = 'NOT_RUN_NO_LEGACY_SOURCE'
    records, timings, status = [], [], 'FAIL'
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for generation in range(args.generations):
            if generation and source:
                with torch.no_grad():
                    for parameter in experts.values():
                        parameter.mul_(0.5)
            wanted_digest = harness.gather(parameter_digest(experts) if source else None)[location]
            for concurrency in CONCURRENCIES:
                selected = request_rows[location:concurrency:4]
                inputs = tokenizer([row['prompt'] for row in selected],
                                   return_tensors='pt',
                                   padding=True,
                                   truncation=True,
                                   max_length=args.max_prompt_tokens)
                inputs = {name: inputs[name].to(device) for name in ('input_ids', 'attention_mask')}
                trace = make_reference(model, inputs, keep_logits, source,
                                       dist.get_rank() + 4 if source else dist.get_rank() - 4, device,
                                       args.decode_tokens)
                for name, route in routes.items():
                    if not source:
                        with torch.no_grad():
                            for parameter in experts.values():
                                parameter.fill_(float('nan'))
                    get_accelerator().synchronize()
                    # The first target use has no intervening global barrier, digest, or reference copy.
                    outputs, tokens, sample = run_cycle(route, model, trace, keep_logits, source, device, True)
                    report = {
                        **sample, 'generation': generation,
                        'concurrency': concurrency,
                        'route': name,
                        'local_active_requests': len(selected),
                        'decode_tokens': args.decode_tokens
                    }
                    if not source:
                        report.update(validate_outputs(outputs, trace, device))
                        if parameter_digest(experts) != wanted_digest:
                            raise AssertionError('the delivered expert weights differ bitwise from the frozen source')
                    del outputs, tokens
                    if parameter_digest(frozen) != frozen_digest:
                        raise AssertionError('an untransferred model parameter or persistent buffer changed')
                    if any((id(parameter), parameter.data_ptr()) != identities_before[key]
                           for key, parameter in experts.items()):
                        raise AssertionError('the model replaced expert Parameter identity or storage during handoff')
                    records.append({'case': f'g{generation}/C{concurrency}/{name}', 'ranks': harness.gather(report)})
                if args.scope == 'paired-performance' and generation == args.generations - 1:
                    timings.extend(
                        paired_windows(args, harness, routes, model, trace, keep_logits, source, device, concurrency))
                    if not source and parameter_digest(experts) != wanted_digest:
                        raise AssertionError('measured repeated handoffs changed the frozen expert snapshot')
                    if parameter_digest(frozen) != frozen_digest:
                        raise AssertionError('measured repeated handoffs changed an untransferred weight')
        status = 'PASS'
    finally:
        if dist.get_rank() == 0:
            import transformers
            result = {
                'status': status,
                'scope': args.scope,
                'records': records,
                'timings': timings,
                'paired_summary': paired_summary(timings),
                'legacy': legacy,
                'unsupported': unsupported,
                'concurrencies': CONCURRENCIES,
                'generations': args.generations,
                'decode_tokens': args.decode_tokens,
                'full_checkpoint_parameters': original_parameters,
                'model_revision': args.model_manifest['revision'],
                'model_manifest_sha256': args.model_manifest_sha256,
                'request_trace_sha256': args.requests_sha256,
                'consumer': 'finite concurrent cohorts; source AutoEP full-model reference; target greedy feedback',
                'teacher_forcing': False,
                'eos_policy': 'fixed decode budget without early EOS termination',
                'handoff_scope': 'expert parameters only; all remaining weights frozen',
                'source_sha256': file_sha256(Path(__file__)),
                'load_and_normalize_seconds': load_seconds,
                'torch': torch.__version__,
                'cuda': torch.version.cuda,
                'transformers': transformers.__version__,
                'h100_count': 8,
                'physical_devices_private': identities,
                'performance_measurement': 'PASS' if status == 'PASS' and timings else 'NOT_RUN',
                'performance_gate': 'NOT_EVALUATED',
                'expansion_admitted': False
            }
            (args.output / 'full_model.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({
                'status': status,
                'full_model_concurrency_correctness': status,
                'performance_gate': 'NOT_EVALUATED',
                'expansion_admitted': False
            }),
                  flush=True)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

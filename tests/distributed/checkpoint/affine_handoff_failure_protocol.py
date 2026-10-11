# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Test real two-rank failure returns separately from the unchanged scientific case matrix."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
from datetime import timedelta

import torch
import deepspeed.comm as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode',
                        choices=('local-failure', 'phase-mismatch', 'oversized', 'nccl-timeout', 'subgroup-timeout'),
                        required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('actual_handoff_harness', args.harness)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    rank = int(os.environ['RANK'])
    args.output.mkdir(parents=True, exist_ok=True)
    harness.JOURNAL = args.output / f'rank-{rank:02d}.jsonl'
    harness.JOURNAL.open('x').close()
    device, _ = harness.initialize(120)
    if dist.get_world_size() != 2:
        raise RuntimeError('failure qualification requires exactly two admitted H100 workers')
    started = time.time()
    observed = None
    collective_elapsed = None
    try:
        if args.mode == 'local-failure':

            def fail_on_target():
                if rank == 1:
                    raise RuntimeError('injected target-local consumer failure')

            try:
                harness.coordinated('consumer-generation-0', fail_on_target)
            except RuntimeError as error:
                observed = str(error)
            assert observed and 'injected target-local consumer failure' in observed
            assert all(harness.gather(True, phase='both-ranks-observed-consumer-failure'))
            tensor = torch.ones(1, device=device)
            dist.all_reduce(tensor)
            assert tensor.item() == 2
        elif args.mode == 'phase-mismatch':
            try:
                harness.gather(None, phase=f'different-phase-{rank}')
            except RuntimeError as error:
                observed = str(error)
            assert observed and 'control phases disagree' in observed and harness.CONTROL_BROKEN
        elif args.mode == 'oversized':
            try:
                harness.gather('x' * (harness.MAX_CONTROL_BYTES + 1) if rank == 1 else 'small', phase='size-limit')
            except RuntimeError as error:
                observed = str(error)
            assert observed and 'bounded length agreement failed' in observed and harness.CONTROL_BROKEN
        else:
            # Establish NCCL on both ranks before deliberately withholding one participant.
            # Keep the other worker alive without entering this collective. The measured
            # return must come from the group's timeout, not from a peer disappearing.
            tensor = torch.ones(1, device=device)
            group = dist.new_group(ranks=[0, 1], timeout=timedelta(
                seconds=120)) if args.mode == 'subgroup-timeout' else None
            dist.all_reduce(tensor, group=group)
            assert tensor.item() == 2
            harness.gather(True, phase='nccl-established')
            outcome_path = args.output / 'data-timeout-outcome.json'
            if rank == 0:
                collective_started = time.monotonic()
                try:
                    # A CUDA collective may return after enqueueing work. An explicit
                    # Work timeout tests host-observed completion and aborts failed work.
                    work = dist.all_reduce(tensor, group=group, async_op=True)
                    work.wait(timeout=timedelta(seconds=120))
                except RuntimeError as error:
                    observed = str(error)
                collective_elapsed = time.monotonic() - collective_started
                temporary = outcome_path.with_suffix('.tmp')
                temporary.write_text(json.dumps({'error': observed, 'elapsed': collective_elapsed}))
                temporary.replace(outcome_path)
            else:
                wait_until = time.monotonic() + 150
                while not outcome_path.exists():
                    if time.monotonic() >= wait_until:
                        raise RuntimeError('data collective did not return within the declared qualification bound')
                    time.sleep(0.5)
                outcome = json.loads(outcome_path.read_text())
                observed, collective_elapsed = outcome['error'], outcome['elapsed']
            assert observed, 'the missing participant must fail, not produce a successful collective'
            assert 110 <= collective_elapsed <= 140, collective_elapsed
        result = {
            'mode': args.mode,
            'rank': rank,
            'status': 'EXPECTED_FAILURE_RETURNED',
            'started': started,
            'finished': time.time(),
            'observed_error': observed,
            'scientific_pass': False,
            'collective_timeout_seconds': 120,
            'collective_elapsed_seconds': collective_elapsed,
            'role': 'observes missing-peer timeout' if rank == 1 and collective_elapsed is not None else 'worker',
            'nccl_blocking_wait': os.environ.get('TORCH_NCCL_BLOCKING_WAIT')
        }
        (args.output / f'rank-{rank:02d}.qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

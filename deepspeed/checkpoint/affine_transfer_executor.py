# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Move a parameter between two shardings by executing a plan from ``affine_transfer``.

The plan names, for each copy, a region of one source shard and the region of one target shard that
should receive it. This module turns those names into data movement: pack out of the source buffer,
across a transport group, through a staging buffer, and into the target buffer. The full logical
parameter is never formed anywhere, which was the point of planning instead of converting.

Three things are deliberately not decided here.

*Endpoints.* Which process actually holds a plan location is supplied by the caller. The plan's two
rank namespaces are not one world size: a source rank 0 has no claim to be the process that is target
rank 0, and folding the two together is how a transfer reads one rank's shard as another's.

*Budget.* How many bytes may be in flight is the caller's, because what fits depends on what else is
resident on the device. A budget that is exhausted is an error, never a silently larger transfer.

*Dtype.* Passed in rather than read from the buffers, because a process that holds no buffer at all
still has to reach the same conclusion as the ones that do -- see ``plan_digest``.

The protocol is: agree on a digest of the schedule before any communication, then work in rounds. In
each round every rank arms all of its receives first and only then posts its sends, so two ranks that
each have something for the other cannot both be waiting on the other's send. That ordering is the
entire deadlock argument, and it only holds because the digest made every rank compute the same
rounds. Returns only once every payload it posted has landed, so a caller can read the target buffers
as soon as it is given control.

Single node, one device per process, synchronous on return. No CUDA graphs, no overlap with compute,
no custom kernels: those all change the schedule, which is the last thing a first version of a data
mover should be perturbing.
"""

import sys
import zlib
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from deepspeed.accelerator import get_accelerator

from .affine_transfer import TransferPlan, TransferSegment

__all__ = [
    'AffineTransferExecutionError', 'TransferEndpoints', 'StagingBudget', 'TransferStats', 'plan_digest',
    'execute_transfer'
]


class AffineTransferExecutionError(ValueError):
    """Wrong enough that no data should have moved."""


@dataclass(frozen=True)
class TransferEndpoints:
    """Which process holds each location named by the plan. ``source`` covers source ranks, ``target``
    covers target ranks, and the two are separate mappings for the reason given above."""
    source: Mapping[int, int]
    target: Mapping[int, int]


@dataclass(frozen=True)
class StagingBudget:
    """Bytes a rank may stage per peer, in each direction, at one time."""
    bytes_per_peer: int = 8 << 20


@dataclass(frozen=True)
class TransferStats:
    segments: int
    chunks: int
    rounds: int
    local_copy_bytes: int
    sent_bytes: int
    received_bytes: int


@dataclass(frozen=True)
class _Operation:
    """One chunk of one segment, resolved to a pair of processes."""
    segment: TransferSegment
    chunk: int
    chunk_shape: Tuple[int, ...]
    source_offset: int
    target_offset: int
    from_process: int
    to_process: int
    elements: int

    @property
    def key(self) -> Tuple:
        return (self.segment.target_rank, self.segment.target_offset, self.segment.source_rank,
                self.segment.source_offset, self.chunk)

    @property
    def payload_bytes(self) -> int:
        return self.elements * self._element_size

    _element_size: int = 0


def _row_major_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def _chunks_per_segment(segment: TransferSegment, budget: StagingBudget, element_size: int) -> int:
    """How many pieces one segment becomes, decided by geometry and budget alone.

    Splitting the last axis keeps every chunk addressable by each side's own strides; splitting any
    other would need strides recomputed, and a chunk that is not a sub-box of its segment is how an
    executor writes outside the plan. The packed cost is the span the chunk covers, not its element
    count, because a column block is strided and packing it touches every row it spans.
    """
    if not segment.shape:
        raise AffineTransferExecutionError(f'{segment!r}: a zero-dimensional segment cannot be transferred')
    last_extent = segment.shape[-1]
    span = max(segment.source_strides[-2], last_extent) if len(segment.shape) > 1 else last_extent
    per_chunk = max(1, min(last_extent, budget.bytes_per_peer // max(span * element_size, 1)))
    return (last_extent + per_chunk - 1) // per_chunk


def _chunk_of(segment: TransferSegment, index: int, count: int, budget: StagingBudget,
              element_size: int) -> TransferSegment:
    last = len(segment.shape) - 1
    extent = segment.shape[last]
    per_chunk = (extent + count - 1) // count
    low = index * per_chunk
    length = min(per_chunk, extent - low)
    shape = segment.shape[:last] + (length, )
    return TransferSegment(source_rank=segment.source_rank,
                           source_offset=segment.source_offset + low * segment.source_strides[last],
                           source_strides=segment.source_strides,
                           target_rank=segment.target_rank,
                           target_offset=segment.target_offset + low * segment.target_strides[last],
                           target_strides=segment.target_strides,
                           shape=shape)


def build_operations(plan: TransferPlan, endpoints: TransferEndpoints, budget: StagingBudget,
                     element_size: int) -> List[_Operation]:
    """The complete schedule, as a list every rank derives identically."""
    operations: List[_Operation] = []
    for segment in plan.segments:
        count = _chunks_per_segment(segment, budget, element_size)
        for index in range(count):
            chunk = _chunk_of(segment, index, count, budget, element_size)
            operation = _Operation(segment=segment,
                                   chunk=index,
                                   chunk_shape=chunk.shape,
                                   source_offset=chunk.source_offset,
                                   target_offset=chunk.target_offset,
                                   from_process=endpoints.source[segment.source_rank],
                                   to_process=endpoints.target[segment.target_rank],
                                   elements=chunk.numel,
                                   _element_size=element_size)
            operations.append(operation)
    operations.sort(key=lambda operation: operation.key)
    return operations


def plan_digest(plan: TransferPlan, endpoints: TransferEndpoints, budget: StagingBudget, element_size: int) -> int:
    """A checksum over the schedule: geometry, addressing, endpoints and budget.

    Deliberately not over buffer contents -- those differ by design. This is the thing every rank must
    be able to compute before it owns any data, which is why dtype is an argument and not an
    inspection.
    """
    payload = [(segment.source_rank, segment.target_rank, segment.source_offset, segment.target_offset, segment.shape,
                segment.source_strides, segment.target_strides) for segment in plan.segments]
    framed = {
        'schema': 2,
        'segments': payload,
        'logical_shape': plan.logical_shape,
        'source_endpoints': dict(sorted(endpoints.source.items())),
        'target_endpoints': dict(sorted(endpoints.target.items())),
        'bytes_per_peer': budget.bytes_per_peer,
        'element_size': element_size,
    }
    return zlib.crc32(repr(sorted(framed.items(), key=lambda item: item[0])).encode())


def _validate_buffers(plan: TransferPlan, source_buffers: Mapping[int, torch.Tensor],
                      target_buffers: Mapping[int, torch.Tensor], dtype: torch.dtype, device: torch.device,
                      endpoints: TransferEndpoints) -> None:
    for location, buffer in list(source_buffers.items()) + list(target_buffers.items()):
        if buffer.dtype != dtype:
            raise AffineTransferExecutionError(f'location {location}: buffer is {buffer.dtype}, plan expects {dtype}')
        if buffer.device != device:
            raise AffineTransferExecutionError(
                f'location {location}: buffer is on {buffer.device}, transport is on {device}')
    for location in source_buffers:
        if location not in endpoints.source:
            raise AffineTransferExecutionError(f'source location {location} has no endpoint binding')
    for location in target_buffers:
        if location not in endpoints.target:
            raise AffineTransferExecutionError(f'target location {location} has no endpoint binding')

    # Storage identity is compared by address range, not by the identity of a storage object: a
    # fresh wrapper is returned per call, so id() would report no overlap even for one buffer.
    source_ranges = [_data_range(buffer) for buffer in source_buffers.values()]
    target_ranges = [_data_range(buffer) for buffer in target_buffers.values()]
    if any(low_a < high_b and low_b < high_a for low_a, high_a in source_ranges for low_b, high_b in target_ranges):
        raise AffineTransferExecutionError(
            'a source and a target share storage; this executor does not order within-storage overlap')

    for segment in plan.segments:
        source = source_buffers.get(segment.source_rank)
        target = target_buffers.get(segment.target_rank)
        for buffer, offset, strides, which in ((source, segment.source_offset, segment.source_strides, 'source'),
                                               (target, segment.target_offset, segment.target_strides, 'target')):
            if buffer is None:
                continue
            reach = offset
            for size, stride in zip(segment.shape, strides):
                if size <= 0:
                    raise AffineTransferExecutionError(f'{which} of {segment!r}: non-positive extent')
                if stride < 0:
                    raise AffineTransferExecutionError(f'{which} of {segment!r}: negative stride')
                if size > 1:
                    reach += (size - 1) * stride
            limit = buffer.numel()
            if reach >= limit:
                raise AffineTransferExecutionError(
                    f'{which} of {segment!r} reaches element {reach} of a {limit}-element buffer')


def execute_transfer(plan: TransferPlan,
                     source_buffers: Mapping[int, torch.Tensor],
                     target_buffers: Mapping[int, torch.Tensor],
                     endpoints: TransferEndpoints,
                     dtype: torch.dtype,
                     budget: StagingBudget = StagingBudget(),
                     group: Optional[object] = None) -> TransferStats:
    """Run every copy of ``plan`` that this process is one end of; block until all have landed.

    ``source_buffers`` and ``target_buffers`` are keyed by the plan's location ranks and need contain
    only what this process holds -- a location missing here must exist at the process its endpoint
    binding names. Buffers are never resized or re-laid-out: a shard that is a window into a bigger
    allocation is addressed by its own storage offset, because making it contiguous would copy the
    whole shard, which is the cost this module exists to avoid.
    """
    import deepspeed.comm as dist

    element_size = torch.tensor(0, dtype=dtype).element_size()
    if element_size == 0:
        raise AffineTransferExecutionError(f'{dtype} has no element size')

    if dist.is_initialized():
        rank = dist.get_rank(group=group)
        device = torch.device(get_accelerator().current_device_name())
    else:
        rank = 0
        device = torch.device('cpu')

    # A rank that cannot validate the transfer must still reach the exchange below: raising here would
    # strand every peer in a collective that never completes. The reason is reported where it was
    # found, and the verdict travels together with the digest.
    try:
        _validate_buffers(plan, source_buffers, target_buffers, dtype, device, endpoints)
        operations = build_operations(plan, endpoints, budget, element_size)
        rejected = 0
    except AffineTransferExecutionError as error:
        print(f'[transfer] rank {rank} rejected the plan: {error}', file=sys.stderr, flush=True)
        operations = []
        rejected = 1

    if dist.is_initialized() and dist.get_world_size(group=group) > 1:
        # A fixed-size integer all_gather, not all_gather_object: the latter pickles its payload and
        # runs several collectives to move a handful of bytes, which measured as a multi-millisecond
        # floor under every transfer regardless of its size.
        digest = plan_digest(plan, endpoints, budget, element_size)
        local = torch.tensor([digest, rejected], dtype=torch.int64, device=device)
        seen = [torch.zeros_like(local) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(seen, local, group=group)
        digests = {int(pair[0].item()) for pair in seen}
        if any(int(pair[1].item()) for pair in seen):
            raise AffineTransferExecutionError(
                'a rank rejected the transfer before it began; no data moved. The reason was reported '
                'by that rank.')
        if len(digests) != 1:
            raise AffineTransferExecutionError(
                f'ranks disagree about the transfer schedule ({sorted(digests)}); refusing to communicate')

    incoming: Dict[int, List[_Operation]] = {}
    outgoing: Dict[int, List[_Operation]] = {}
    for operation in operations:
        if operation.to_process == rank and operation.from_process != rank:
            incoming.setdefault(operation.from_process, []).append(operation)
        elif operation.from_process == rank and operation.to_process != rank:
            outgoing.setdefault(operation.to_process, []).append(operation)

    arenas: Dict[int, torch.Tensor] = {}
    send_arenas: Dict[int, torch.Tensor] = {}
    for peer, peer_operations in sorted(incoming.items()):
        elements = max(operation.elements for operation in peer_operations)
        if elements * element_size > budget.bytes_per_peer:
            raise AffineTransferExecutionError(
                f'peer {peer}: a single staged chunk needs {elements * element_size} bytes over a budget '
                f'of {budget.bytes_per_peer}')
        arenas[peer] = torch.empty(elements, dtype=dtype, device=device)
    for peer, peer_operations in sorted(outgoing.items()):
        send_arenas[peer] = torch.empty(max(operation.elements for operation in peer_operations),
                                        dtype=dtype,
                                        device=device)

    rounds = max((len(peer_operations) for peer_operations in list(incoming.values()) + list(outgoing.values())),
                 default=0)
    if not operations:
        return TransferStats(0, 0, 0, 0, 0, 0)

    sent = received = copied = 0

    for round_index in range(rounds):
        staged, requests = [], []
        for peer, peer_operations in sorted(incoming.items()):
            if round_index >= len(peer_operations):
                continue
            operation = peer_operations[round_index]
            buffer = arenas[peer][:operation.elements]
            requests.append(('recv', buffer, peer))
            staged.append((buffer, operation))

        for peer, peer_operations in sorted(outgoing.items()):
            if round_index >= len(peer_operations):
                continue
            operation = peer_operations[round_index]
            source = _flat(operation.segment.source_rank, source_buffers, dtype)
            payload = send_arenas[peer][:operation.elements]
            payload.view(operation.chunk_shape).copy_(
                _region(source, operation.source_offset, operation.chunk_shape, operation.segment.source_strides))
            requests.append(('send', payload, peer))
            staged.append((payload, operation))

        # One batched stage per round, not one op at a time. On a process group of more than one rank
        # torch treats an individually posted P2P op as a collective, so a rank's lone irecv would be
        # matched against the peer's lone irecv and both would wait forever; a batch carries the send
        # and the receive together, which is what makes "arm receives before sends" actually hold.
        handles = dist.batch_p2p(requests, group=group)
        for handle in handles:
            handle.wait()

        for buffer, operation in staged:
            if operation.to_process == rank and operation.from_process != rank:
                target = _flat(operation.segment.target_rank, target_buffers, dtype)
                _region(target, operation.target_offset, operation.chunk_shape,
                        operation.segment.target_strides).copy_(buffer.view(operation.chunk_shape))
                received += operation.payload_bytes
            elif operation.from_process == rank and operation.to_process != rank:
                sent += operation.payload_bytes

    for operation in operations:
        if operation.from_process != rank or operation.to_process != rank:
            continue
        source = _flat(operation.segment.source_rank, source_buffers, dtype)
        target = _flat(operation.segment.target_rank, target_buffers, dtype)
        _region(target, operation.target_offset, operation.chunk_shape, operation.segment.target_strides).copy_(
            _region(source, operation.source_offset, operation.chunk_shape, operation.segment.source_strides))
        copied += operation.payload_bytes

    return TransferStats(segments=len(plan.segments),
                         chunks=len(operations),
                         rounds=rounds,
                         local_copy_bytes=copied,
                         sent_bytes=sent,
                         received_bytes=received)


def _data_range(buffer: torch.Tensor) -> Tuple[int, int]:
    """Byte extent a tensor's elements occupy in storage."""
    start = buffer.data_ptr()
    span = buffer.numel() * buffer.element_size()
    offset = buffer.storage_offset() * buffer.element_size()
    return start - offset, start - offset + buffer.untyped_storage().nbytes()


def _flat(location: int, buffers: Mapping[int, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """The buffer for ``location`` as its own elements, without copying it."""
    try:
        buffer = buffers[location]
    except KeyError:
        raise AffineTransferExecutionError(f'location {location} is bound to this process but holds no buffer')
    flat = buffer.reshape(-1)
    if flat.stride(0) != 1:
        raise AffineTransferExecutionError(f'location {location}: shard is not contiguous in its own storage')
    if flat.dtype != dtype:
        raise AffineTransferExecutionError(f'location {location}: {flat.dtype} is not {dtype}')
    return flat


def _region(flat: torch.Tensor, offset: int, shape: Tuple[int, ...], strides: Tuple[int, ...]) -> torch.Tensor:
    """``flat`` viewed as one planned copy region.

    Plan offsets count elements from the first element of the shard; ``as_strided`` counts from the
    start of the storage. Adding the buffer's own storage offset is the explicit form of that, and the
    alternative -- making the buffer contiguous -- would copy the whole shard to fix arithmetic.
    """
    return torch.as_strided(flat, size=shape, stride=strides, storage_offset=flat.storage_offset() + offset)

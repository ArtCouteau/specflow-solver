from __future__ import annotations

import torch
import torch.distributed as dist

from .. import context


def _rank_and_world_size() -> tuple[int, int]:
    distribution = context.get().distribution
    return distribution.rank, distribution.worldSize


def _split_sizes(globalSize: int, worldSize: int) -> list[int]:
    base, remainder = divmod(globalSize, worldSize)
    sizes = [base] * worldSize
    for rank in range(worldSize - remainder, worldSize):
        sizes[rank] += 1
    return sizes


def local_size(globalSize: int) -> int:
    rank, worldSize = _rank_and_world_size()
    return _split_sizes(globalSize, worldSize)[rank]


def local_slice(globalSize: int) -> tuple[int, int]:
    rank, worldSize = _rank_and_world_size()
    sizes = _split_sizes(globalSize, worldSize)
    start = sum(sizes[:rank])
    return start, start + sizes[rank]


def transpose(
    tensor: torch.Tensor,
    splitAxis: int,
    splitAxisGlobalSize: int,
    gatherAxis: int,
    gatherAxisGlobalSize: int,
) -> torch.Tensor:
    """Redistribute `tensor` from split-along-`gatherAxis` (full along `splitAxis`)
    to split-along-`splitAxis` (full along `gatherAxis`) via one all-to-all."""
    rank, worldSize = _rank_and_world_size()
    if worldSize == 1:
        return tensor

    splitAxisSizes = _split_sizes(splitAxisGlobalSize, worldSize)
    gatherAxisSizes = _split_sizes(gatherAxisGlobalSize, worldSize)

    # `splitAxis` is currently full locally (size splitAxisGlobalSize); chop it into
    # per-destination-rank chunks. `gatherAxis` is currently split (local size
    # gatherAxisSizes[rank]) and is left untouched by this split.
    sendChunks = [chunk.contiguous() for chunk in tensor.split(splitAxisSizes, dim=splitAxis)]

    recvShape = list(tensor.shape)
    recvShape[splitAxis] = splitAxisSizes[rank]
    recvChunks = []
    for source in range(worldSize):
        shape = list(recvShape)
        shape[gatherAxis] = gatherAxisSizes[source]
        recvChunks.append(torch.empty(shape, dtype=tensor.dtype, device=tensor.device))

    dist.all_to_all(recvChunks, sendChunks)

    return torch.cat(recvChunks, dim=gatherAxis)


def mirror_exchange(localTensor: torch.Tensor) -> torch.Tensor:
    """Swap this rank's whole local slab with rank `worldSize - 1 - rank` (its fixed
    mirror partner). Self-exchange (no-op) when a rank is its own partner."""
    rank, worldSize = _rank_and_world_size()
    partner = worldSize - 1 - rank
    if partner == rank:
        return localTensor.clone()

    localTensor = localTensor.contiguous()
    received = torch.empty_like(localTensor)
    ops = [
        dist.P2POp(dist.isend, localTensor, partner),
        dist.P2POp(dist.irecv, received, partner),
    ]
    for request in dist.batch_isend_irecv(ops):
        request.wait()

    return received


def gather_full(localTensor: torch.Tensor, splitAxis: int, splitAxisGlobalSize: int) -> torch.Tensor:
    """Collect every rank's local slab onto rank 0 only, concatenated along `splitAxis`.
    Implemented with point-to-point send/recv rather than a collective gather, since
    gather/scatter support is not guaranteed across `torch.distributed` backends."""
    rank, worldSize = _rank_and_world_size()
    if worldSize == 1:
        return localTensor

    if rank != 0:
        dist.send(localTensor.contiguous(), dst=0)
        return localTensor

    sizes = _split_sizes(splitAxisGlobalSize, worldSize)
    chunks = [localTensor]
    for source in range(1, worldSize):
        shape = list(localTensor.shape)
        shape[splitAxis] = sizes[source]
        chunk = torch.empty(shape, dtype=localTensor.dtype, device=localTensor.device)
        dist.recv(chunk, src=source)
        chunks.append(chunk)

    return torch.cat(chunks, dim=splitAxis)


def scatter_full(
    fullTensor: torch.Tensor | None,
    localShape: tuple[int, ...],
    splitAxis: int,
    splitAxisGlobalSize: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    """Inverse of `gather_full`: rank 0 holds the full tensor and sends each rank its
    local slab. `fullTensor` is ignored on every rank other than 0."""
    rank, worldSize = _rank_and_world_size()
    if worldSize == 1:
        assert fullTensor is not None
        return fullTensor

    if rank != 0:
        localTensor = torch.empty(localShape, dtype=dtype, device=device)
        dist.recv(localTensor, src=0)
        return localTensor

    assert fullTensor is not None
    sizes = _split_sizes(splitAxisGlobalSize, worldSize)
    chunks = list(fullTensor.split(sizes, dim=splitAxis))
    for destination in range(1, worldSize):
        dist.send(chunks[destination].contiguous(), dst=destination)

    return chunks[0].contiguous()

from __future__ import annotations

import torch
import torch.distributed as dist

from .. import context


def _rank_and_world_size() -> tuple[int, int]:
    distribution = context.get().distribution
    return distribution.rank, distribution.world_size


def _split_sizes(global_size: int, world_size: int) -> list[int]:
    base, remainder = divmod(global_size, world_size)
    sizes = [base] * world_size
    for rank in range(world_size - remainder, world_size):
        sizes[rank] += 1
    return sizes


def local_size(global_size: int) -> int:
    rank, world_size = _rank_and_world_size()
    return _split_sizes(global_size, world_size)[rank]


def local_slice(global_size: int) -> tuple[int, int]:
    rank, world_size = _rank_and_world_size()
    sizes = _split_sizes(global_size, world_size)
    start = sum(sizes[:rank])
    return start, start + sizes[rank]


def transpose(
    tensor: torch.Tensor,
    split_axis: int,
    split_axis_global_size: int,
    gather_axis: int,
    gather_axis_global_size: int,
) -> torch.Tensor:
    """Redistribute `tensor` from split-along-`gather_axis` (full along `split_axis`)
    to split-along-`split_axis` (full along `gather_axis`) via one all-to-all."""
    rank, world_size = _rank_and_world_size()
    if world_size == 1:
        return tensor

    split_axis_sizes = _split_sizes(split_axis_global_size, world_size)
    gather_axis_sizes = _split_sizes(gather_axis_global_size, world_size)

    send_chunks = [chunk.contiguous() for chunk in tensor.split(gather_axis_sizes, dim=gather_axis)]

    recv_shape = list(tensor.shape)
    recv_shape[gather_axis] = gather_axis_sizes[rank]
    recv_chunks = []
    for source in range(world_size):
        shape = list(recv_shape)
        shape[split_axis] = split_axis_sizes[source]
        recv_chunks.append(torch.empty(shape, dtype=tensor.dtype, device=tensor.device))

    dist.all_to_all(recv_chunks, send_chunks)

    return torch.cat(recv_chunks, dim=split_axis)


def mirror_exchange(local_tensor: torch.Tensor) -> torch.Tensor:
    """Swap this rank's whole local slab with rank `world_size - 1 - rank` (its fixed
    mirror partner). Self-exchange (no-op) when a rank is its own partner."""
    rank, world_size = _rank_and_world_size()
    partner = world_size - 1 - rank
    if partner == rank:
        return local_tensor.clone()

    local_tensor = local_tensor.contiguous()
    received = torch.empty_like(local_tensor)
    ops = [
        dist.P2POp(dist.isend, local_tensor, partner),
        dist.P2POp(dist.irecv, received, partner),
    ]
    for request in dist.batch_isend_irecv(ops):
        request.wait()

    return received


def gather_full(local_tensor: torch.Tensor, split_axis: int, split_axis_global_size: int) -> torch.Tensor:
    """Collect every rank's local slab onto rank 0 only, concatenated along `split_axis`.
    Implemented with point-to-point send/recv rather than a collective gather, since
    gather/scatter support is not guaranteed across `torch.distributed` backends."""
    rank, world_size = _rank_and_world_size()
    if world_size == 1:
        return local_tensor

    if rank != 0:
        dist.send(local_tensor.contiguous(), dst=0)
        return local_tensor

    sizes = _split_sizes(split_axis_global_size, world_size)
    chunks = [local_tensor]
    for source in range(1, world_size):
        shape = list(local_tensor.shape)
        shape[split_axis] = sizes[source]
        chunk = torch.empty(shape, dtype=local_tensor.dtype, device=local_tensor.device)
        dist.recv(chunk, src=source)
        chunks.append(chunk)

    return torch.cat(chunks, dim=split_axis)


def scatter_full(
    full_tensor: torch.Tensor | None,
    local_shape: tuple[int, ...],
    split_axis: int,
    split_axis_global_size: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    """Inverse of `gather_full`: rank 0 holds the full tensor and sends each rank its
    local slab. `full_tensor` is ignored on every rank other than 0."""
    rank, world_size = _rank_and_world_size()
    if world_size == 1:
        assert full_tensor is not None
        return full_tensor

    if rank != 0:
        local_tensor = torch.empty(local_shape, dtype=dtype, device=device)
        dist.recv(local_tensor, src=0)
        return local_tensor

    assert full_tensor is not None
    sizes = _split_sizes(split_axis_global_size, world_size)
    chunks = list(full_tensor.split(sizes, dim=split_axis))
    for destination in range(1, world_size):
        dist.send(chunks[destination].contiguous(), dst=destination)

    return chunks[0].contiguous()

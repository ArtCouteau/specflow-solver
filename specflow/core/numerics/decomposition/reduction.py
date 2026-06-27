from __future__ import annotations

import torch
import torch.distributed as dist

from .. import context

Dim = int | tuple[int, ...] | None


def _world_size() -> int:
    return context.get().distribution.worldSize


def global_sum(tensor: torch.Tensor, dim: Dim = None) -> torch.Tensor:
    result = tensor.sum() if dim is None else tensor.sum(dim=dim)
    if _world_size() > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def global_mean(tensor: torch.Tensor, dim: Dim = None) -> torch.Tensor:
    if dim is None:
        localSum = tensor.sum()
        localCount = tensor.numel()
    else:
        localSum = tensor.sum(dim=dim)
        axes = dim if isinstance(dim, tuple) else (dim,)
        localCount = 1
        for axis in axes:
            localCount *= tensor.shape[axis]

    if _world_size() == 1:
        return localSum / localCount

    count = torch.tensor(localCount, dtype=localSum.dtype, device=localSum.device)
    dist.all_reduce(localSum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return localSum / count


def global_max(tensor: torch.Tensor, dim: Dim = None) -> torch.Tensor:
    if dim is None:
        result = tensor.max()
    else:
        result = tensor.amax(dim=dim)
    if _world_size() > 1:
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result

from __future__ import annotations

import torch
import torch.distributed as dist

from .. import context

Dim = int | tuple[int, ...] | None


def _world_size() -> int:
    return context.get().distribution.world_size


def global_sum(tensor: torch.Tensor, dim: Dim = None) -> torch.Tensor:
    result = tensor.sum() if dim is None else tensor.sum(dim=dim)
    if _world_size() > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def global_mean(tensor: torch.Tensor, dim: Dim = None) -> torch.Tensor:
    if dim is None:
        local_sum = tensor.sum()
        local_count = tensor.numel()
    else:
        local_sum = tensor.sum(dim=dim)
        axes = dim if isinstance(dim, tuple) else (dim,)
        local_count = 1
        for axis in axes:
            local_count *= tensor.shape[axis]

    if _world_size() == 1:
        return local_sum / local_count

    count = torch.tensor(local_count, dtype=local_sum.dtype, device=local_sum.device)
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return local_sum / count


def global_max(tensor: torch.Tensor, dim: Dim = None) -> torch.Tensor:
    if dim is None:
        result = tensor.max()
    else:
        result = tensor.amax(dim=dim)
    if _world_size() > 1:
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result

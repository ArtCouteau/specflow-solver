import torch

from .. import context, dims
from ...physics import variable


def zeros(rank: variable.Rank) -> torch.Tensor:
    return torch.zeros(
        size=variable.field_shape(rank),
        dtype=context.get().config.get('numerics/precision/complex', torch.dtype),
        device=context.get().config.get('numerics/device', str)
    )


def transpose(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(dims.TENSOR[0], dims.TENSOR[1])

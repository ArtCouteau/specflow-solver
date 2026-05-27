import torch

from . import norm
from .. import context


def shell(lower: float, upper: float) -> torch.Tensor:
    return (
        (norm.square_l2_norm(context.get().mesh.grid) > lower**2)
        & (norm.square_l2_norm(context.get().mesh.grid) <= upper**2)
    )


def spherical(upper: float) -> torch.Tensor:
    return norm.square_l2_norm(context.get().mesh.grid) <= upper**2

import torch

from . import norm
from .. import mesh
from ... import context


def shell(lower: float, upper: float) -> torch.Tensor:
    return (
        (norm.square_l2_norm(context.get().mesh.grid) > lower**2)
        & (norm.square_l2_norm(context.get().mesh.grid) <= upper**2)
    )


def full_shell(lower: float, upper: float) -> torch.Tensor:
    _fullGrid = mesh.construct_full_spectral_grid()
    return (
        (norm.square_l2_norm(_fullGrid) > lower**2)
        & (norm.square_l2_norm(_fullGrid) <= upper**2)
    )


def spherical(upper: float) -> torch.Tensor:
    return norm.square_l2_norm(context.get().mesh.grid) <= upper**2

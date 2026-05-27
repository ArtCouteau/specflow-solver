import torch

from . import norm
from .. import context


def spherical(lower: float | None = None, upper: float | None = None) -> torch.Tensor:
    if lower is None or lower <= 0.0:
        lower = 0.0
    if upper is None:
        upper = float(context.get().config.get('grid/N', int))

    return (
        (norm.square_l2_norm(context.get().mesh.grid) > lower**2)
        & (norm.square_l2_norm(context.get().mesh.grid) <= upper**2)
    )

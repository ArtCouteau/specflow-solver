import torch

from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Tensor, TSpace

from ... import context
from ...physics import variable


def zeros(rank: variable.Rank) -> torch.Tensor:
    return torch.zeros(
        size=variable.field_shape(rank),
        dtype=context.get().config.get("numerics/precision/complex", torch.dtype),
        device=context.get().config.get("numerics/device", str),
    )


def transpose(field: Field[Tensor, TSpace]) -> Field[Tensor, TSpace]:
    return field.with_data(field.data.transpose(field.component_axes[0], field.component_axes[1]))

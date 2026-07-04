import torch

from ....core.numerics.operations import volume
from ....core.physics import fields


def residual_dissipation_rate(velocity: torch.Tensor, eddyViscosity: torch.Tensor) -> torch.Tensor:
    return volume.mean(
        fields.physical.les.residual_dissipation(velocity, eddyViscosity)
    )

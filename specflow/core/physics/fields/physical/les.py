import torch

from ....numerics.operations import fourier, norm
from ... import fields


def residual_dissipation(velocity: torch.Tensor, eddyViscosity: torch.Tensor) -> torch.Tensor:
    return (
        2.0
        * fourier.to_physical(eddyViscosity)
        * norm.square_frobenius_norm(fields.physical.velocity.strain_rate_tensor(velocity))
    )

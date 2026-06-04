import torch

from ....numerics.operations import derivative, norm, rank


def vorticity(velocity: torch.Tensor) -> torch.Tensor:
    return derivative.curl_vector(velocity)


def strain_rate_tensor(velocity: torch.Tensor) -> torch.Tensor:
    return 0.5*(
        derivative.gradient_vector(velocity)
        + rank.transpose(derivative.gradient_vector(velocity))
    )


def rotation_rate_tensor(velocity: torch.Tensor) -> torch.Tensor:
    return 0.5*(
        derivative.gradient_vector(velocity)
        - rank.transpose(derivative.gradient_vector(velocity))
    )


def modal_kinetic_energy(velocity: torch.Tensor) -> torch.Tensor:
    return 0.5*norm.square_l2_norm(velocity)

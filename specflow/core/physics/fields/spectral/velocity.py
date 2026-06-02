import torch

from ....numerics.operations import derivative, norm


def vorticity(velocity: torch.Tensor) -> torch.Tensor:
    return derivative.curl_vector(velocity)


def modal_kinetic_energy(velocity: torch.Tensor) -> torch.Tensor:
    return 0.5*norm.square_l2_norm(velocity)

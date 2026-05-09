import torch

from ....numerics.operations import derivative


def vorticity(velocity: torch.Tensor) -> torch.Tensor:
    return derivative.curl_vector(velocity)

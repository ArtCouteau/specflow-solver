import torch
from typing import Callable

from . import algebra, norm
from .. import context


Gradient = Callable[[torch.Tensor], torch.Tensor]
Laplacian = Callable[[torch.Tensor], torch.Tensor]


def nabla() -> torch.Tensor:
    return (
        1.0j
        * context.get().mesh.grid
    ).to(context.get().config.get('numerics/precision/complex', torch.dtype))


def nabla2() -> torch.Tensor:
    return (
        - norm.square_l2_norm(context.get().mesh.grid)
    ).to(context.get().config.get('numerics/precision/float', torch.dtype))


def inv_nabla2() -> torch.Tensor:
    return (
        - 1.0/torch.where(
            norm.square_l2_norm(context.get().mesh.grid) == 0,
            1,
            norm.square_l2_norm(context.get().mesh.grid)
        )
    ).to(context.get().config.get('numerics/precision/float', torch.dtype))


def curl_vector(vector: torch.Tensor) -> torch.Tensor:
    return algebra.cross_vector_vector(nabla(), vector)


def divergence_vector(vector: torch.Tensor) -> torch.Tensor:
    return algebra.inner_vector_vector(nabla(), vector)


def divergence_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return algebra.inner_vector_tensor(nabla(), tensor)


def gradient_scalar(scalar: torch.Tensor) -> torch.Tensor:
    return nabla()*scalar


def gradient_vector(vector: torch.Tensor) -> torch.Tensor:
    return algebra.outer_vector_vector(nabla(), vector)


def laplacian_scalar(scalar: torch.Tensor) -> torch.Tensor:
    return nabla2()*scalar


def laplacian_vector(vector: torch.Tensor) -> torch.Tensor:
    return nabla2()*vector

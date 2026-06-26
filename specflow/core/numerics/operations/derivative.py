import torch
from typing import Callable

from . import algebra
from .. import context


Gradient = Callable[[torch.Tensor], torch.Tensor]
Laplacian = Callable[[torch.Tensor], torch.Tensor]


class DerivativeOperators:
    nabla: torch.Tensor
    nabla2: torch.Tensor
    invNabla2: torch.Tensor

    def __init__(self):
        self.nabla = (
            1.0j * context.get().mesh.k
        ).to(context.get().config.get('numerics/precision/complex', torch.dtype))

        self.nabla2 = (
            - context.get().mesh.k2
        ).to(context.get().config.get('numerics/precision/float', torch.dtype))

        self.invNabla2 = (
            - 1.0/torch.where(context.get().mesh.k2 == 0, 1, context.get().mesh.k2)
        ).to(context.get().config.get('numerics/precision/float', torch.dtype))


def nabla() -> torch.Tensor:
    return context.get().physics.derivative.nabla


def nabla2() -> torch.Tensor:
    return context.get().physics.derivative.nabla2


def inv_nabla2() -> torch.Tensor:
    return context.get().physics.derivative.invNabla2


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


from typing import Callable
import torch

from . import rank
from .. import dims

InterRankOperation = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def cross_vector_vector(vectorA: torch.Tensor, vectorB: torch.Tensor) -> torch.Tensor:
    return vectorA.cross(vectorB, dim=dims.VECTOR[0])


def inner_vector_vector(vectorA: torch.Tensor, vectorB: torch.Tensor) -> torch.Tensor:
    return (vectorA*vectorB).sum(dim=dims.VECTOR)


def inner_vector_tensor(vector: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return (vector*rank.transpose(tensor)).sum(dim=1)


def inner_tensor_tensor(tensorA: torch.Tensor, tensorB: torch.Tensor) -> torch.Tensor:
    return (tensorA*tensorB).sum(dim=dims.TENSOR)


def outer_vector_vector(vectorA: torch.Tensor, vectorB: torch.Tensor) -> torch.Tensor:
    return vectorA.unsqueeze(1)*vectorB.unsqueeze(0)


def mul_scalar_vector(scalar: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return scalar*vector


def mul_scalar_tensor(scalar: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return scalar*tensor


def mul_tensor_tensor(tensorA: torch.Tensor, tensorB: torch.Tensor) -> torch.Tensor:
    return tensorA*tensorB


def matmul_tensor_vector(tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (tensor*vector).sum(dim=1)

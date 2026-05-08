import torch

from ...core.defs import DIMS_VECTOR, DIMS_TENSOR


def square_l2_norm_scalar(data: torch.Tensor) -> torch.Tensor:
    return (data.abs()**2)


def l2_norm(data: torch.Tensor) -> torch.Tensor:
    return square_l2_norm(data).sqrt()


def square_l2_norm(data: torch.Tensor) -> torch.Tensor:
    return (data.abs()**2).sum(dim=DIMS_VECTOR)


def square_frobenius_norm(data: torch.Tensor) -> torch.Tensor:
    return (data.abs()**2).sum(dim=DIMS_TENSOR)

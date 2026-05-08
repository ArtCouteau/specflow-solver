import torch

from .. import dims


def transpose(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(dims.TENSOR[0], dims.TENSOR[1])

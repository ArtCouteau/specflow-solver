import torch

from .. import dims


def sum(field: torch.Tensor) -> torch.Tensor:
    return field.sum(dim=dims.domain())


def mean(field: torch.Tensor) -> torch.Tensor:
    return field.mean(dim=dims.domain())

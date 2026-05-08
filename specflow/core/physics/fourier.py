import torch

from . import dims


def to_physical(data: torch.Tensor) -> torch.Tensor:
    return torch.fft.irfftn(data, dim=dims.domain(), norm='forward')  # type: ignore


def to_spectral(data: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfftn(data, dim=dims.domain(), norm='forward')  # type: ignore

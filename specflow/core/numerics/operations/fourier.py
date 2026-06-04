import torch

from .. import dims


def to_physical(data: torch.Tensor) -> torch.Tensor:
    return torch.fft.irfftn(data, dim=dims.domain(), norm='forward')  # type: ignore


def to_spectral(data: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfftn(data, dim=dims.domain(), norm='forward')  # type: ignore


def to_full_spectral(data: torch.Tensor) -> torch.Tensor:
    # Assumes that the data is half spectral
    return torch.fft.fftn(to_physical(data), dim=dims.domain(), norm='forward')  # type: ignore

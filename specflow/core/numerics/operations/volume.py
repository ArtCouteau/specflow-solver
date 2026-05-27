import torch

from .. import dims


def sum(field: torch.Tensor) -> torch.Tensor:
    return field.sum(dim=dims.domain())


def sum_over_spectral_domain(spectralField: torch.Tensor) -> torch.Tensor:
    return (
        spectralField.sum(dim=dims.domain())
        + spectralField[..., 1:].sum(dim=dims.domain())
    )


def mean(field: torch.Tensor) -> torch.Tensor:
    return field.mean(dim=dims.domain())

import torch

from .. import dims
from ... import context


def sum(field: torch.Tensor) -> torch.Tensor:
    return field.sum(dim=dims.domain())


def sum_over_spectral_domain(spectralField: torch.Tensor) -> torch.Tensor:
    paired = spectralField[..., 1:]
    if context.get().config.get('grid/N', int) % 2 == 0:
        paired = paired[..., :-1]

    return (
        spectralField.sum(dim=dims.domain())
        + paired.sum(dim=dims.domain())
    )


def mean(field: torch.Tensor) -> torch.Tensor:
    return field.mean(dim=dims.domain())

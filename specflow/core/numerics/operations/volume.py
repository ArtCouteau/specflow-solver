import torch

from .. import context, dims
from ..decomposition import reduction


def sum(field: torch.Tensor) -> torch.Tensor:
    return reduction.global_sum(field, dim=dims.domain())


def sum_over_spectral_domain(spectralField: torch.Tensor) -> torch.Tensor:
    paired = spectralField[..., 1:]
    if context.get().config.get('grid/N', int) % 2 == 0:
        paired = paired[..., :-1]

    return (
        reduction.global_sum(spectralField, dim=dims.domain())
        + reduction.global_sum(paired, dim=dims.domain())
    )


def mean(field: torch.Tensor) -> torch.Tensor:
    return reduction.global_mean(field, dim=dims.domain())

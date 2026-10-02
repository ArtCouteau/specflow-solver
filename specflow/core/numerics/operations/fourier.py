import torch

from specflow.core.physics.quantity.field import Field, Physical, Spectral, TRank


def to_physical(field: Field[TRank, Spectral]) -> Field[TRank, Physical]:
    return Field(
        torch.fft.irfftn(field.data, dim=field.domain_axes, norm="forward"), field.rank, Physical, field.dimension
    )


def to_spectral(field: Field[TRank, Physical]) -> Field[TRank, Spectral]:
    return Field(
        torch.fft.rfftn(field.data, dim=field.domain_axes, norm="forward"), field.rank, Spectral, field.dimension
    )

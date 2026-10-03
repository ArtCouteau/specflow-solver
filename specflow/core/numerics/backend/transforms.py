from abc import ABC, abstractmethod
from typing import override

import torch

from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Physical, Spectral, TRank


class TransformsBackend(ABC):
    @abstractmethod
    def rfftn(self, field: Field[TRank, Physical]) -> Field[TRank, Spectral]: ...

    @abstractmethod
    def irfftn(self, field: Field[TRank, Spectral]) -> Field[TRank, Physical]: ...


class SerialTransforms(TransformsBackend):
    @override
    def rfftn(self, field: Field[TRank, Physical]) -> Field[TRank, Spectral]:
        return Field(
            torch.fft.rfftn(field.data, dim=field.domain_axes, norm="forward"),
            field.rank,
            Spectral,
            field.component,
            field.dimension,
        )

    @override
    def irfftn(self, field: Field[TRank, Spectral]) -> Field[TRank, Physical]:
        return Field(
            torch.fft.irfftn(field.data, dim=field.domain_axes, norm="forward"),
            field.rank,
            Physical,
            field.component,
            field.dimension,
        )

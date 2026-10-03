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
        return field.replace(data=torch.fft.rfftn(field.data, dim=field.domain_axes, norm="forward"), space=Spectral)

    @override
    def irfftn(self, field: Field[TRank, Spectral]) -> Field[TRank, Physical]:
        return field.replace(data=torch.fft.irfftn(field.data, dim=field.domain_axes, norm="forward"), space=Physical)

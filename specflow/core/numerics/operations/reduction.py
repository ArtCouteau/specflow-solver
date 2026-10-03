from abc import ABC, abstractmethod
from typing import override

from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import TRank, TSpace
from specflow.core.physics.quantity.statistics import IntegralValue


class ReductionBackend(ABC):
    @abstractmethod
    def sum(self, field: Field[TRank, TSpace]) -> IntegralValue[TRank]: ...

    @abstractmethod
    def mean(self, field: Field[TRank, TSpace]) -> IntegralValue[TRank]: ...


class SerialReduction(ReductionBackend):
    @override
    def sum(self, field: Field[TRank, TSpace]) -> IntegralValue[TRank]:
        return IntegralValue(field.data.sum(field.domain_axes), field.rank)

    @override
    def mean(self, field: Field[TRank, TSpace]) -> IntegralValue[TRank]:
        return IntegralValue(field.data.mean(field.domain_axes), field.rank)

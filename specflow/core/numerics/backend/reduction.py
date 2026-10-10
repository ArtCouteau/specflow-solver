from abc import ABC, abstractmethod
from typing import override

from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, TRank, TSpace
from specflow.processing.physics.statistics.wrappers import IntegralValue


class ReductionBackend(ABC):
    @abstractmethod
    def global_sum(self, field: Field[TRank, TSpace]) -> IntegralValue[TRank]: ...

    @abstractmethod
    def global_mean(self, field: Field[TRank, Physical]) -> IntegralValue[TRank]: ...

    @abstractmethod
    def global_max(self, field: Field[TRank, Physical]) -> float: ...


class SerialReduction(ReductionBackend):
    @override
    def global_sum(self, field: Field[TRank, TSpace]) -> IntegralValue[TRank]:
        return IntegralValue(field.data.sum(field.domain_axes), field.rank)

    @override
    def global_mean(self, field: Field[TRank, Physical]) -> IntegralValue[TRank]:
        return IntegralValue(field.data.mean(field.domain_axes), field.rank)

    @override
    def global_max(self, field: Field[TRank, Physical]) -> float:
        return field.data.max().item()

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

from specflow.core.physics.wrappers.statistics import IntegralValue
from specflow.core.physics.wrappers.wrapper import Physical, TRank, TSpace

if TYPE_CHECKING:
    from specflow.core.physics.wrappers.field import Field


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

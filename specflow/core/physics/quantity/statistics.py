from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Quantity, Rank, TRank, TSpace

R = TypeVar("R", bound=Rank)


@dataclass(frozen=True, eq=False)
class IntegralValue(Quantity, Generic[TRank]):
    rank: type[TRank]

    @staticmethod
    def from_field(data: torch.Tensor, field: Field[R, TSpace]) -> IntegralValue[R]:
        return IntegralValue(data, field.rank)

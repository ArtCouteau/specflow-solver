from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from specflow.core.physics.quantity.quantity import Quantity, TRank


@dataclass(frozen=True, eq=False)
class IntegralValue(Quantity, Generic[TRank]):
    rank: type[TRank]

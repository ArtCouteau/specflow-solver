from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from specflow.core.physics.quantity.quantity import Quantity


class Rank(enum.Enum):
    SCALAR = 0
    VECTOR = 1
    TENSOR = 2


class Space(enum.Enum):
    PHYSICAL = 0
    SPECTRAL = 1


TRank = TypeVar("TRank", bound=Rank)
TSpace = TypeVar("TSpace", bound=Space)


@dataclass(frozen=True, eq=False)
class Field(Quantity, Generic[TRank, TSpace]):
    rank: type[TRank]
    space: type[TSpace]

    def __add__(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        return self.replace_data(self.data + other.data)

    def __radd__(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        return self.__add__(other)

    def __sub__(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        return self.replace_data(self.data - other.data)

    def __rsub__(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        return self.__sub__(other)

    def __neg__(self) -> Field[TRank, TSpace]:
        return self.replace_data(-self.data)

    def __mul__(self, c: float) -> Field[TRank, TSpace]:
        return self.replace_data(c * self.data)

    def __rmul__(self, c: float) -> Field[TRank, TSpace]:
        return self.__mul__(c)

    def replace_data(self, data: torch.Tensor) -> Field[TRank, TSpace]:
        return Field(data, self.rank, self.space)

    def add_(self, other: Field[TRank, TSpace]) -> None:
        self.data.add_(other.data)

    def copy_(self, other: Field[TRank, TSpace]) -> None:
        self.data.copy_(other.data)

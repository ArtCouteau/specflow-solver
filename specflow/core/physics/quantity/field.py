from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import torch

from specflow.core.physics.quantity.quantity import Quantity


class Rank: ...


class Scalar(Rank): ...


class Vector(Rank): ...


class Tensor(Rank): ...


class Space: ...


class Physical(Space): ...


class Spectral(Space): ...


TRank = TypeVar("TRank", bound=Rank)
TSpace = TypeVar("TSpace", bound=Space)


@dataclass(frozen=True, eq=False)
class Field(Quantity, Generic[TRank, TSpace]):
    rank: type[TRank]
    space: type[TSpace]
    dimension: Literal[1, 2, 3]

    @property
    def component_axes(self: Field[Vector, TSpace] | Field[Tensor, TSpace]) -> tuple[int, ...]:
        if self.rank is Vector:
            return (0,)
        elif self.rank is Tensor:
            return (0, 1)
        else:
            raise TypeError(f"Field of rank {self.rank} has unknown component_axes")

    @property
    def domain_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.dimension, 0))

    def __add__(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        return self.with_data(self.data + other.data)

    def __radd__(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        return self.__add__(other)

    def __sub__(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        return self.with_data(self.data - other.data)

    def __rsub__(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        return self.__sub__(other)

    def __neg__(self) -> Field[TRank, TSpace]:
        return self.with_data(-self.data)

    def __mul__(self, c: float) -> Field[TRank, TSpace]:
        return self.with_data(c * self.data)

    def __rmul__(self, c: float) -> Field[TRank, TSpace]:
        return self.__mul__(c)

    def with_data(self, data: torch.Tensor) -> Field[TRank, TSpace]:
        return Field(data, self.rank, self.space, self.dimension)

    def add_(self, other: Field[TRank, TSpace]) -> None:
        self.data.add_(other.data)

    def copy_(self, other: Field[TRank, TSpace]) -> None:
        self.data.copy_(other.data)

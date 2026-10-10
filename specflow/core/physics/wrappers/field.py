from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypedDict, TypeVar, Unpack, overload

import torch

from specflow.core.numerics.backend.backend import NumericsBackend
from specflow.core.numerics.mesh import Mesh
from specflow.core.physics.wrappers.wrapper import (
    Rank,
    Scalar,
    Space,
    Tensor,
    TRank,
    TSpace,
    Vector,
    Wrapper,
)

TNewRank = TypeVar("TNewRank", bound=Rank)
TNewSpace = TypeVar("TNewSpace", bound=Space)


class FieldChanges(TypedDict, total=False):
    data: torch.Tensor
    component: Literal[1, 2, 3]
    dimension: Literal[1, 2, 3]
    backend: NumericsBackend
    mesh: Mesh


@dataclass(frozen=True, eq=False)
class Field(Wrapper, Generic[TRank, TSpace]):
    rank: type[TRank]
    space: type[TSpace]
    component: Literal[1, 2, 3]
    dimension: Literal[1, 2, 3]
    backend: NumericsBackend = field(repr=False)
    mesh: Mesh = field(repr=False)

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
        return self.replace(data=self.data + other.data)

    def __radd__(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        return self.__add__(other)

    def __sub__(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        return self.replace(data=self.data - other.data)

    def __rsub__(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        return self.__sub__(other)

    def __neg__(self) -> Field[TRank, TSpace]:
        return self.replace(data=-self.data)

    def __mul__(self, c: complex) -> Field[TRank, TSpace]:
        return self.replace(data=c * self.data)

    def __rmul__(self, c: complex) -> Field[TRank, TSpace]:
        return self.__mul__(c)

    @overload
    def replace(self, **changes: Unpack[FieldChanges]) -> Field[TRank, TSpace]: ...

    @overload
    def replace(self, *, rank: type[TNewRank], **changes: Unpack[FieldChanges]) -> Field[TNewRank, TSpace]: ...

    @overload
    def replace(self, *, space: type[TNewSpace], **changes: Unpack[FieldChanges]) -> Field[TRank, TNewSpace]: ...

    @overload
    def replace(
        self, *, rank: type[TNewRank], space: type[TNewSpace], **changes: Unpack[FieldChanges]
    ) -> Field[TNewRank, TNewSpace]: ...

    def replace(self, **changes: Any) -> Field[Any, Any]:
        if changes.get("rank", self.rank) is Scalar:
            changes["component"] = 1
        return dataclasses.replace(self, **changes)

    def copy_(self, other: Field[TRank, TSpace]) -> Field[TRank, TSpace]:
        self.data.copy_(other.data)
        return self

    def add_(self, other: Field[TRank, TSpace], alpha: float | None = None) -> Field[TRank, TSpace]:
        self.data.add_(other.data, alpha=alpha)
        return self

    def mul_(self, other: complex) -> Field[TRank, TSpace]:
        self.data.mul_(other)
        return self

    def sqrt_(self) -> Field[TRank, TSpace]:
        self.data.sqrt_()
        return self

    def pow_(self, exponent: float) -> Field[TRank, TSpace]:
        self.data.pow_(exponent)
        return self

    def zero_(self) -> Field[TRank, TSpace]:
        self.data.zero_()
        return self

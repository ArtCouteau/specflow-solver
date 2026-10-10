from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from specflow.core.physics.wrappers.wrapper import TRank, Wrapper

if TYPE_CHECKING:
    from specflow.processing.numerics.shell import SpectralShells


class Measure: ...


class Density(Measure): ...


class Accumulation(Measure): ...


TMeasure = TypeVar("TMeasure", bound=Measure)


@dataclass(frozen=True, eq=False)
class IntegralValue(Wrapper, Generic[TRank]):
    rank: type[TRank]


@dataclass(frozen=True, eq=False)
class Spectrum(Wrapper, Generic[TRank, TMeasure]):
    rank: type[TRank]
    measure: type[TMeasure]
    shells: SpectralShells = field(repr=False)

    def cumulate(self: Spectrum[TRank, Density]) -> Spectrum[TRank, Accumulation]:
        return Spectrum(self.data.cumsum(dim=-1), self.rank, Accumulation, self.shells)


@dataclass(frozen=True, eq=False)
class Slice(Wrapper, Generic[TRank]):
    rank: type[TRank]
    axis: int
    index: int


TQuantity = TypeVar("TQuantity", bound=Wrapper)


@dataclass(frozen=True, eq=False)
class Series(Generic[TQuantity]):
    data: torch.Tensor
    template: TQuantity = field(repr=False)

    @staticmethod
    def stack(items: Sequence[TQuantity]) -> Series[TQuantity]:
        return Series(torch.stack([q.data for q in items]), items[0])

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, index: int) -> TQuantity:
        return dataclasses.replace(self.template, data=self.data[index])

    def mean(self) -> TQuantity:
        return dataclasses.replace(self.template, data=self.data.mean(dim=0))

    def var(self) -> TQuantity:
        return dataclasses.replace(self.template, data=self.data.var(dim=0, correction=0))

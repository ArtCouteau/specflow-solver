from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin

from specflow.core.numerics.operations import fourier
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.statistics import Accumulation, Density, IntegralValue, Spectrum
from specflow.core.physics.wrappers.wrapper import Physical, Spectral, TRank, Wrapper
from specflow.processing.numerics.kind import substitute, unify
from specflow.processing.numerics.shell import spectral_shells

TIn = TypeVar("TIn", bound=Wrapper)
TOut = TypeVar("TOut", bound=Wrapper)


class Operator(Generic[TIn, TOut], ABC):
    # Maps a node to the node of same name with another kind. The kinds come from the generic base of the subclass,
    # where a TypeVar (e.g. TRank) matches any type.
    source: ClassVar[Any]
    target: ClassVar[Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is Operator:
                cls.source, cls.target = get_args(base)

    @abstractmethod
    def apply(self, value: TIn) -> TOut: ...

    def source_kind(self, target: Any) -> Any | None:
        bindings = unify(self.target, target)
        return None if bindings is None else substitute(self.source, bindings)


class ToPhysical(Operator[Field[TRank, Spectral], Field[TRank, Physical]]):
    def apply(self, value: Field[TRank, Spectral]) -> Field[TRank, Physical]:
        return fourier.to_physical(value)


class ToSpectral(Operator[Field[TRank, Physical], Field[TRank, Spectral]]):
    def apply(self, value: Field[TRank, Physical]) -> Field[TRank, Spectral]:
        return fourier.to_spectral(value)


class VolumeMean(Operator[Field[TRank, Physical], IntegralValue[TRank]]):
    def apply(self, value: Field[TRank, Physical]) -> IntegralValue[TRank]:
        return value.backend.reduction.global_mean(value)


class ZeroMode(Operator[Field[TRank, Spectral], IntegralValue[TRank]]):
    # With forward-normalized transforms, the zero mode is the volume mean
    def apply(self, value: Field[TRank, Spectral]) -> IntegralValue[TRank]:
        return IntegralValue(value.data[(..., *(0,) * value.dimension)].real, value.rank)


class ShellSpectrum(Operator[Field[TRank, Spectral], Spectrum[TRank, Density]]):
    def apply(self, value: Field[TRank, Spectral]) -> Spectrum[TRank, Density]:
        shells = spectral_shells(value.mesh)
        weighted = value.data * value.mesh.spectral.parsevalWeights.data
        return Spectrum(shells.sum(weighted, value.dimension), value.rank, Density, shells)


class Cumulate(Operator[Spectrum[TRank, Density], Spectrum[TRank, Accumulation]]):
    def apply(self, value: Spectrum[TRank, Density]) -> Spectrum[TRank, Accumulation]:
        return value.cumulate()


OPERATORS: tuple[Operator[Any, Any], ...] = (
    ToPhysical(),
    ToSpectral(),
    VolumeMean(),
    ZeroMode(),
    ShellSpectrum(),
    Cumulate(),
)

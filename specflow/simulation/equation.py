from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, override

from specflow.core.numerics.operations import fourier
from specflow.core.numerics.operations.projection import Projector
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, Scalar, Spectral, TRank, Vector
from specflow.simulation.terms.term import Term

TRankEquation = TypeVar("TRankEquation", bound=Scalar | Vector)


@dataclass(frozen=True)
class Equation(Generic[TRank]):
    tag: str
    rank: type[TRank]
    terms: dict[str, Term[TRank]]
    rhs: Field[TRank, Spectral]
    variable: Field[TRank, Spectral]

    def compute_rhs(self, time: float) -> Field[TRank, Spectral]:
        self.rhs.zero_()
        for term in self.terms.values():
            term.compute(self.rhs, time)

        return self.rhs

    def set_initial_conditions_physical(self, field: Field[TRank, Physical]) -> None:
        self.variable.copy_(fourier.to_spectral(field))

    def set_initial_conditions_spectral(self, field: Field[TRank, Spectral]) -> None:
        self.variable.copy_(field)


@dataclass(frozen=True)
class NavierStokesEquation(Equation[Vector]):
    projector: Projector

    @override
    def compute_rhs(self, time: float) -> Field[Vector, Spectral]:
        return self.rhs.copy_(self.projector.apply(self.variable.mesh, super().compute_rhs(time)))

from __future__ import annotations

from dataclasses import dataclass

from specflow.core.numerics.operations import algebra, derivative
from specflow.core.physics import fields
from specflow.core.physics.dealias import QuadraticDealias
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, Scalar, Spectral, Tensor, TRank, Vector
from specflow.simulation.terms.term import Term, TermRegistry


@dataclass(frozen=True)
class AdvectionTerm(Term[TRank]):
    advecting: Field[Vector, Spectral]
    advected: Field[TRank, Spectral]
    dealias: QuadraticDealias


@TermRegistry.register("Advection/Rotation")
@dataclass(frozen=True)
class RotationFormVectorAdvectionTerm(AdvectionTerm[Vector]):
    def _physical_space_operation(
        self, velocity: Field[Vector, Physical], vorticity: Field[Vector, Physical]
    ) -> Field[Vector, Physical]:
        return -algebra.cross_vector_vector(vorticity, velocity)

    def compute(self, rhs: Field[Vector, Spectral], time: float) -> None:
        rhs.add_(
            self.dealias.apply(
                self._physical_space_operation,
                self.advecting,
                fields.spectral.velocity.vorticity(self.advected),
            )
        )


@TermRegistry.register("Advection/ConvectionScalar")
@dataclass(frozen=True)
class ConvectionFormScalarAdvectionTerm(AdvectionTerm[Scalar]):
    def _physical_space_operation(
        self, velocity: Field[Vector, Physical], gradient: Field[Vector, Physical]
    ) -> Field[Scalar, Physical]:
        return -algebra.inner_vector_vector(velocity, gradient)

    def compute(self, rhs: Field[Scalar, Spectral], time: float) -> None:
        rhs.add_(
            self.dealias.apply(
                self._physical_space_operation,
                self.advecting,
                derivative.gradient_scalar(self.advected),
            )
        )


@TermRegistry.register("Advection/ConvectionVector")
@dataclass(frozen=True)
class ConvectionFormVectorAdvectionTerm(AdvectionTerm[Vector]):
    def _physical_space_operation(
        self, velocity: Field[Vector, Physical], gradient: Field[Tensor, Physical]
    ) -> Field[Vector, Physical]:
        return -algebra.inner_vector_tensor(velocity, gradient)

    def compute(self, rhs: Field[Vector, Spectral], time: float) -> None:
        rhs.add_(
            self.dealias.apply(
                self._physical_space_operation,
                self.advecting,
                derivative.gradient_vector(self.advected),
            )
        )

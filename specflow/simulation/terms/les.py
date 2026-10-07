from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from specflow.core.numerics.operations import algebra, derivative
from specflow.core.physics import fields
from specflow.core.physics.dealias import FourierSmoothing
from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Physical, Scalar, Spectral, Tensor, Vector
from specflow.simulation.terms.term import Term, TermRegistry


@dataclass(frozen=True)
class LESModelTerm(Term[Vector]):
    smoothing: FourierSmoothing
    velocity: Field[Vector, Spectral]

    @abstractmethod
    def _eddy_viscosity(self, strainRateTensor: Field[Tensor, Physical]) -> Field[Scalar, Physical]: ...

    def _stress_tensor(self, strainRateTensor: Field[Tensor, Physical]) -> Field[Tensor, Physical]:
        return 2.0 * algebra.scale(strainRateTensor, self._eddy_viscosity(strainRateTensor))

    def compute(self, rhs: Field[Vector, Spectral], time: float) -> None:
        rhs.add_(
            derivative.divergence_tensor(
                self.smoothing.apply(self._stress_tensor, fields.spectral.velocity.strain_rate_tensor(self.velocity))
            )
        )


@TermRegistry.register("LES/Smagorinsky")
@dataclass(frozen=True)
class SmagorinskyLESModelTerm(LESModelTerm):
    filterWidth: float
    constant: float = 0.12

    def _eddy_viscosity(self, strainRateTensor: Field[Tensor, Physical]) -> Field[Scalar, Physical]:
        return (self.constant * self.filterWidth) ** 2 * fields.physical.velocity.rate_of_strain(strainRateTensor)

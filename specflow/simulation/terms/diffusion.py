from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, override

from specflow.core.numerics.operations import algebra, derivative
from specflow.core.physics import fields
from specflow.core.physics.dealias import FourierSmoothing
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, Scalar, Spectral, Tensor, TRank, Vector
from specflow.processing.numerics.rule import Rule, derive
from specflow.simulation.terms.term import Term, TermRegistry


@dataclass(frozen=True)
class DiffusionTerm(Term[TRank]):
    diffused: Field[TRank, Spectral]


@dataclass(frozen=True)
class ConstantDiffusivityDiffusionTerm(DiffusionTerm[TRank]):
    diffusivity: float


@TermRegistry.register("Diffusion/ConstantScalar")
@dataclass(frozen=True)
class ConstantDiffusivityScalarDiffusionTerm(ConstantDiffusivityDiffusionTerm[Scalar]):
    def compute(self, rhs: Field[Scalar, Spectral], time: float) -> None:
        rhs.add_(self.diffusivity * derivative.laplacian_scalar(self.diffused))


@TermRegistry.register("Diffusion/ConstantVector")
@dataclass(frozen=True)
class ConstantDiffusivityVectorDiffusionTerm(ConstantDiffusivityDiffusionTerm[Vector]):
    def compute(self, rhs: Field[Vector, Spectral], time: float) -> None:
        rhs.add_(self.diffusivity * derivative.laplacian_vector(self.diffused))


@TermRegistry.register("Diffusion/HyperViscousVector")
@dataclass(frozen=True)
class HyperViscousVectorDiffusionTerm(ConstantDiffusivityDiffusionTerm[Vector]):
    exponent: int

    def compute(self, rhs: Field[Vector, Spectral], time: float) -> None:
        rhs.add_(
            self.diffusivity
            * algebra.scale(self.diffused, derivative.nabla_square(self.diffused.mesh).pow_(self.exponent))
        )


@dataclass(frozen=True)
class GeneralizedNewtonianDiffusionTerm(DiffusionTerm[Vector]):
    smoothing: FourierSmoothing

    @abstractmethod
    def _viscosity(self, strainRateTensor: Field[Tensor, Physical]) -> Field[Scalar, Physical]: ...

    def _stress_tensor(self, strainRateTensor: Field[Tensor, Physical]) -> Field[Tensor, Physical]:
        return 2.0 * algebra.scale(strainRateTensor, self._viscosity(strainRateTensor))

    def compute(self, rhs: Field[Vector, Spectral], time: float) -> None:
        rhs.add_(
            derivative.divergence_tensor(
                self.smoothing.apply(self._stress_tensor, fields.spectral.velocity.strain_rate_tensor(self.diffused))
            )
        )

    def rules(self) -> list[Rule[Any]]:
        return [derive("Viscosity", self._viscosity, "StrainRateTensor")]


@TermRegistry.register("Diffusion/GeneralizedNewtonianPowerLaw")
@dataclass(frozen=True)
class GeneralizedNewtonianPowerLawDiffusionTerm(GeneralizedNewtonianDiffusionTerm):
    baseCoefficient: float
    powerLawExponent: float

    @override
    def _viscosity(self, strainRateTensor: Field[Tensor, Physical]) -> Field[Scalar, Physical]:
        rateOfStrain = fields.physical.velocity.rate_of_strain(strainRateTensor).data
        return self.baseCoefficient * strainRateTensor.replace(
            data=rateOfStrain.abs_().pow_(self.powerLawExponent - 1), rank=Scalar
        )


@TermRegistry.register("Diffusion/GeneralizedNewtonianCarreau")
@dataclass(frozen=True)
class GeneralizedNewtonianCarreauDiffusionTerm(GeneralizedNewtonianDiffusionTerm):
    lowerLimit: float
    upperLimit: float
    consistencyIndex: float
    powerIndex: float

    @override
    def _viscosity(self, strainRateTensor: Field[Tensor, Physical]) -> Field[Scalar, Physical]:
        rateOfStrain = fields.physical.velocity.rate_of_strain(strainRateTensor).data
        return strainRateTensor.replace(
            data=(
                self.upperLimit
                + (self.lowerLimit - self.upperLimit)
                * (1.0 + (self.consistencyIndex * rateOfStrain).pow_(2)).pow_((self.powerIndex - 1.0) / 2.0)
            ),
            rank=Scalar,
        )

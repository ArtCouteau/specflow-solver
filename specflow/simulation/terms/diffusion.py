from __future__ import annotations

import torch
from typing import Callable

from . import term
from .. import solution
from ...core.numerics import context
from ...core.numerics.operations import algebra, derivative, fourier
from ...core.physics import fields, variable


class DiffusionTerm(term.Term):

    def __init__(self, tag: str, equation: str, variable: str, rank: variable.Rank):
        super().__init__(tag, equation, variable, rank)

    def time_step_limiter(self) -> tuple[str, float, Callable[[solution.SolutionVariable, float], float]] | None:
        return (
            f'Diffusion/{self.tag}',
            context.get().config.get(f'equations/{self.equation}/terms/{self.tag}/limiter', float, 0.5),
            self._diffusion_number
        )

    def _diffusion_number(self, variable: solution.SolutionVariable, dt: float) -> float: ...


@term.TermRegistry.register('diffusion/constant')
class ConstantDiffusivityDiffusionTerm(DiffusionTerm):
    diffusivity: float
    laplacian: derivative.Laplacian

    def __init__(self, tag: str, equation: str, variable: str, rank: variable.Rank):
        super().__init__(tag, equation, variable, rank)

        if self.rank.name == 'SCALAR':
            self.laplacian = derivative.laplacian_scalar
        elif self.rank.name == 'VECTOR':
            self.laplacian = derivative.laplacian_vector
        else:
            raise NotImplementedError()

        self.diffusivity = context.get().config.get(f'equations/{self.equation}/terms/{tag}/diffusivity', float)

    def _diffusion_number(self, variable: solution.SolutionVariable, dt: float) -> float:
        return (
            self.diffusivity
            * dt
            / context.get().mesh.physical_grid_spacing()**2
        )

    def compute(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor:
        return self.diffusivity * self.laplacian(variable.get(self.variable))


@term.TermRegistry.register('diffusion/hyperViscous')
class ConstantDiffusivityHyperDiffusionTerm(DiffusionTerm):
    coefficient: float
    exponent: int

    def __init__(self, tag: str, equation: str, variable: str, rank: variable.Rank):
        super().__init__(tag, equation, variable, rank)

        if self.rank.name not in ('SCALAR', 'VECTOR'):
            raise NotImplementedError()

        self.coefficient = context.get().config.get(f'equations/{self.equation}/terms/{tag}/coefficient', float)
        self.exponent = context.get().config.get(f'equations/{self.equation}/terms/{tag}/exponent', int)
        if not (self.exponent > 0 and self.exponent.bit_count() == 1):
            raise ValueError(
                f'Hyper viscous terms only accept power of two exponents (term "{self.tag}" of equation '
                f'"{self.equation}")'
            )

    def _diffusion_number(self, variable: solution.SolutionVariable, dt: float) -> float:
        return (
            self.coefficient
            * dt
            / context.get().mesh.physical_grid_spacing()**(2.0*self.exponent)
        )

    def compute(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor:
        return self.coefficient * derivative.nabla2()**self.exponent * variable.get(self.variable)


class GeneralizedNewtonianDiffusionTerm(DiffusionTerm):
    def __init__(self, tag: str, equation: str, variable: str, rank: variable.Rank):
        super().__init__(tag, equation, variable, rank)

        if self.rank.name == 'SCALAR':
            self.laplacian = derivative.laplacian_scalar
        elif self.rank.name == 'VECTOR':
            self.laplacian = derivative.laplacian_vector
        else:
            raise NotImplementedError()

        self.diffusivity = context.get().config.get(f'equations/{self.equation}/terms/{tag}/diffusivity', float)

    def _diffusion_number(self, variable: solution.SolutionVariable, dt: float) -> float:
        return (
            self._physical_viscosity(
                fields.physical.velocity.strain_rate_tensor(variable.get('velocity'))
            ).max().item()
            * dt
            / context.get().mesh.physical_grid_spacing()**2
        )

    def _get_viscosity(self, velocity: torch.Tensor) -> torch.Tensor:
        return fourier.to_spectral(
            self._physical_viscosity(
                fields.physical.velocity.strain_rate_tensor(velocity.data)
            )
        )

    def _physical_viscosity(self, physicalStrainRateTensor: torch.Tensor) -> torch.Tensor: ...

    def _physical_stress_tensor(self, physicalStrainRateTensor: torch.Tensor) -> torch.Tensor:
        return 2.0*algebra.mul_scalar_tensor(
            self._physical_viscosity(physicalStrainRateTensor),
            physicalStrainRateTensor
        )

    def diagnostic_fields(self, variable: solution.SolutionVariable, time: float) -> dict[str, torch.Tensor]:
        return {
            'viscosity': fourier.to_spectral(
                self._physical_viscosity(
                    fields.spectral.velocity.strain_rate_tensor(variable.get('velocity'))
                )
            )
        }

    def compute(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor:
        return derivative.divergence_tensor(
            context.get().physics.dealias.apply(
                self._physical_stress_tensor,
                fields.spectral.velocity.strain_rate_tensor(variable.get('velocity'))
            )
        )


@term.TermRegistry.register('diffusion/generalizedNewtonianPowerLaw')
class GeneralizedNewtonianPowerLawDiffusionTerm(GeneralizedNewtonianDiffusionTerm):
    baseCoefficient: float
    powerLawExponent: float

    def __init__(self, tag: str, equation: str, variable: str, rank: variable.Rank):
        super().__init__(tag, equation, variable, rank)

        self.baseCoefficient = context.get().config.get(f'equations/{self.equation}/terms/{tag}/baseCoefficient', float)
        self.powerLawExponent = context.get().config.get(
            f'equations/{self.equation}/terms/{tag}/powerLawExponent', float)

    def _physical_viscosity(self, physicalStrainRateTensor: torch.Tensor) -> torch.Tensor:
        return self.baseCoefficient*(
            fields.physical.velocity.rate_of_strain(physicalStrainRateTensor).abs().pow(self.powerLawExponent - 1)
        )


@term.TermRegistry.register('diffusion/generalizedNewtonianCarreau')
class GeneralizedNewtonianCarreauDiffusionTerm(GeneralizedNewtonianDiffusionTerm):
    lowerLimit: float
    upperLimit: float
    consistencyIndex: float
    powerIndex: float

    def __init__(self, tag: str, equation: str, variable: str, rank: variable.Rank):
        super().__init__(tag, equation, variable, rank)

        self.lowerLimit = context.get().config.get(f'equations/{self.equation}/terms/{tag}/lowerLimit', float)
        self.upperLimit = context.get().config.get(f'equations/{self.equation}/terms/{tag}/upperLimit', float)
        self.consistencyIndex = context.get().config.get(
            f'equations/{self.equation}/terms/{tag}/consistencyIndex', float)
        self.powerIndex = context.get().config.get(f'equations/{self.equation}/terms/{tag}/powerIndex', float)

    def _physical_viscosity(self, physicalStrainRateTensor: torch.Tensor) -> torch.Tensor:
        return (
            self.upperLimit
            + (self.lowerLimit - self.upperLimit)
            * (
                1.0
                + (self.consistencyIndex * fields.physical.velocity.rate_of_strain(physicalStrainRateTensor)).pow(2)
            ).pow((self.powerIndex - 1.0)/2.0)
        )

from __future__ import annotations

import torch

from . import term
from .. import equation, solution
from ...core.numerics import context
from ...core.numerics.operations import algebra, derivative, fourier
from ...core.physics import fields, variable


class DiffusionTerm(term.Term):

    def __init__(self, equation: equation.Equation, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        self.equation.solver.integrator.add_time_step_limiter(
            f'Diffusion/{tag}',
            context.get().config.get(f'equations/{equation.tag}/terms/{tag}/limiter', float, 0.5),
            self._diffusion_number
        )

    def _diffusion_number(self) -> float: ...


@term.TermRegistry.register('diffusion', 'constant')
class ConstantDiffusivityDiffusionTerm(DiffusionTerm):
    diffusivity: float
    laplacian: derivative.Laplacian

    def __init__(self, equation: equation.Equation, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        if self.rank.name == 'SCALAR':
            self.laplacian = derivative.laplacian_scalar
        elif self.rank.name == 'VECTOR':
            self.laplacian = derivative.laplacian_vector
        else:
            raise NotImplementedError()

        self.diffusivity = context.get().config.get(f'equations/{equation.tag}/terms/{tag}/diffusivity', float)

    def _diffusion_number(self) -> float:
        return (
            self.diffusivity
            * self.equation.solver.integrator.dt
            / context.get().mesh.physical_grid_spacing()**2
        )

    def compute(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor:
        return self.diffusivity * self.laplacian(variable.get(self.equation.variable))


@term.TermRegistry.register('diffusion', 'hyperViscous')
class ConstantDiffusivityHyperDiffusionTerm(DiffusionTerm):
    coefficient: float
    exponent: int
    laplacian: derivative.Laplacian

    def __init__(self, equation: equation.Equation, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        if self.rank.name == 'SCALAR':
            self.laplacian = derivative.laplacian_scalar
        elif self.rank.name == 'VECTOR':
            self.laplacian = derivative.laplacian_vector
        else:
            raise NotImplementedError()

        self.coefficient = context.get().config.get(f'equations/{equation.tag}/terms/{tag}/coefficient', float)
        self.exponent = context.get().config.get(f'equations/{equation.tag}/terms/{tag}/exponent', int)
        if not (self.exponent > 0 and self.exponent.bit_count() == 1):
            raise ValueError(
                f'Hyper viscous terms only accept power of two exponents (term "{self.tag}" of equation '
                f'"{self.equation}")'
            )

    def _diffusion_number(self) -> float:
        return (
            self.coefficient
            * self.equation.solver.integrator.dt
            / context.get().mesh.physical_grid_spacing()**(self.exponent)
        )

    def compute(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor:
        return self.coefficient * self.laplacian(variable.get(self.equation.variable))**(self.exponent//2)


class GeneralizedNewtonianDiffusionTerm(DiffusionTerm):
    velocity: variable.Variable

    def __init__(self, equation: equation.Equation, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        if self.rank.name == 'SCALAR':
            self.laplacian = derivative.laplacian_scalar
        elif self.rank.name == 'VECTOR':
            self.laplacian = derivative.laplacian_vector
        else:
            raise NotImplementedError()

        self.diffusivity = context.get().config.get(f'equations/{equation.tag}/terms/{tag}/diffusivity', float)

    def _diffusion_number(self) -> float:
        return (
            self._physical_viscosity(
                fields.physical.velocity.strain_rate_tensor(self.velocity.data)
            ).max().item()
            * self.equation.solver.integrator.dt
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

    def compute(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor:
        return derivative.divergence_tensor(
            context.get().physics.dealias.apply(
                self._physical_stress_tensor,
                fields.spectral.velocity.strain_rate_tensor(variable.get('velocity'))
            )
        )


@term.TermRegistry.register('diffusion', 'generalizedNewtonianPowerLaw')
class GeneralizedNewtonianPowerLawDiffusionTerm(GeneralizedNewtonianDiffusionTerm):
    baseCoefficient: float
    powerLawExponent: float

    def __init__(self, equation: equation.Equation, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)
        self.baseCoefficient = context.get().config.get(f'equations/{equation.tag}/terms/{tag}/baseCoefficient', float)
        self.powerLawExponent = context.get().config.get(
            f'equations/{equation.tag}/terms/{tag}/powerLawExponent', float)

    def _physical_viscosity(self, physicalStrainRateTensor: torch.Tensor) -> torch.Tensor:
        return self.baseCoefficient*(
            fields.physical.velocity.rate_of_strain(physicalStrainRateTensor).abs().pow(self.powerLawExponent - 1)
        )


@term.TermRegistry.register('diffusion', 'generalizedNewtonianCarreau')
class GeneralizedNewtonianCarreauDiffusionTerm(GeneralizedNewtonianDiffusionTerm):
    lowerLimit: float
    upperLimit: float
    consistencyIndex: float
    powerIndex: float

    def __init__(self, equation: equation.Equation, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)
        self.lowerLimit = context.get().config.get(f'equations/{equation.tag}/terms/{tag}/lowerLimit', float)
        self.upperLimit = context.get().config.get(f'equations/{equation.tag}/terms/{tag}/upperLimit', float)
        self.consistencyIndex = context.get().config.get(
            f'equations/{equation.tag}/terms/{tag}/consistencyIndex', float)
        self.powerIndex = context.get().config.get(f'equations/{equation.tag}/terms/{tag}/powerIndex', float)

    def _physical_viscosity(self, physicalStrainRateTensor: torch.Tensor) -> torch.Tensor:
        return (
            self.upperLimit
            + (self.lowerLimit - self.upperLimit)
            * (
                1.0
                + (self.consistencyIndex * fields.physical.velocity.rate_of_strain(physicalStrainRateTensor)).pow(2)
            ).pow((self.powerIndex - 1.0)/2.0)
        )

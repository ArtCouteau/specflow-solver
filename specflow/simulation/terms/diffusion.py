from __future__ import annotations

import torch

from . import term
from .. import solver
from ...core.numerics import context
from ...core.numerics.operations import algebra, derivative, fourier
from ...core.physics import fields, variable


class DiffusionTerm(term.Term):
    def __init__(self, equation: str, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        context.get().sim.integrator.add_time_step_limiter(
            tag,
            context.get().config.get(f'equations/{equation}/terms/{tag}/limiter', float, 0.5),
            self._diffusion_number
        )

    def _diffusion_number(self) -> float: ...


@term.TermRegistry.register('diffusion', 'constant')
class ConstantDiffusivityDiffusionTerm(DiffusionTerm):
    diffusivity: float
    laplacian: derivative.Laplacian

    def __init__(self, equation: str, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        if self.rank.name == 'SCALAR':
            self.laplacian = derivative.laplacian_scalar
        elif self.rank.name == 'VECTOR':
            self.laplacian = derivative.laplacian_vector
        else:
            raise NotImplementedError()

        self.diffusivity = context.get().config.get(f'equations/{equation}/terms/{tag}/diffusivity', float)

    def _diffusion_number(self) -> float:
        return (
            self.diffusivity
            * context.get().sim.integrator.dt
            / context.get().mesh.physical_grid_spacing()**2
        )

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor:
        return self.diffusivity * self.laplacian(variable)


@term.TermRegistry.register('diffusion', 'hyperViscous')
class ConstantDiffusivityHyperDiffusionTerm(DiffusionTerm):
    coefficient: float
    exponent: int
    laplacian: derivative.Laplacian

    def __init__(self, equation: str, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        if self.rank.name == 'SCALAR':
            self.laplacian = derivative.laplacian_scalar
        elif self.rank.name == 'VECTOR':
            self.laplacian = derivative.laplacian_vector
        else:
            raise NotImplementedError()

        self.coefficient = context.get().config.get(f'equations/{equation}/terms/{tag}/coefficient', float)
        self.exponent = context.get().config.get(f'equations/{equation}/terms/{tag}/exponent', int)
        if not (self.exponent > 0 and self.exponent.bit_count() == 1):
            raise ValueError(
                f'Hyper viscous terms only accept power of two exponents (term "{self.tag}" of equation'
                f'"{self.equation}")'
            )

    def _diffusion_number(self) -> float:
        return (
            self.coefficient
            * context.get().sim.integrator.dt
            / context.get().mesh.physical_grid_spacing()**(self.exponent)
        )

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor:
        return self.coefficient * self.laplacian(variable.data)**(self.exponent//2)


class GeneralizedNewtonianDiffusionTerm(DiffusionTerm):
    velocity: variable.Variable

    def __init__(self, equation: str, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        if self.rank.name == 'SCALAR':
            self.laplacian = derivative.laplacian_scalar
        elif self.rank.name == 'VECTOR':
            self.laplacian = derivative.laplacian_vector
        else:
            raise NotImplementedError()

        self.diffusivity = context.get().config.get(f'equations/{equation}/terms/{tag}/diffusivity', float)

    def _diffusion_number(self) -> float:
        return (
            self._physical_viscosity(
                fields.physical.velocity.strain_rate_tensor(self.velocity.data)
            ).max().item()
            * context.get().sim.integrator.dt
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

    def add_references(self, solver: solver.Solver) -> None:
        _eq = context.get().config.get(f'equations/{self.equation}/terms/{self.tag}/velocity/equation', str)
        if not _eq in solver.equations.keys():
            raise KeyError(
                f'Reference equation for advection term "{self.tag}" of equation "{self.equation}" not found'
            )
        elif not solver.equations[_eq].rank.name == 'VECTOR':
            raise ValueError(
                f'Reference equation for advection term "{self.tag}" of equation "{self.equation}" is not a vector equation'
            )
        else:
            self.velocity = solver.equations[_eq].variable

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor:
        return derivative.divergence_tensor(
            context.get().physics.dealias.apply(
                self._physical_stress_tensor,
                fields.spectral.velocity.strain_rate_tensor(variable)
            )
        )


@term.TermRegistry.register('diffusion', 'generalizedNewtonianPowerLaw')
class GeneralizedNewtonianPowerLawDiffusionTerm(GeneralizedNewtonianDiffusionTerm):
    baseCoefficient: float
    powerLawExponent: float

    def __init__(self, equation: str, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)
        self.baseCoefficient = context.get().config.get(f'equations/{equation}/terms/{tag}/baseCoefficient', float)
        self.powerLawExponent = context.get().config.get(f'equations/{equation}/terms/{tag}/powerLawExponent', float)

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

    def __init__(self, equation: str, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)
        self.lowerLimit = context.get().config.get(f'equations/{equation}/terms/{tag}/lowerLimit', float)
        self.upperLimit = context.get().config.get(f'equations/{equation}/terms/{tag}/upperLimit', float)
        self.consistencyIndex = context.get().config.get(f'equations/{equation}/terms/{tag}/consistencyIndex', float)
        self.powerIndex = context.get().config.get(f'equations/{equation}/terms/{tag}/powerIndex', float)

    def _physical_viscosity(self, physicalStrainRateTensor: torch.Tensor) -> torch.Tensor:
        return (
            self.upperLimit
            + (self.lowerLimit - self.upperLimit)
            * (
                1.0
                + (self.consistencyIndex * fields.physical.velocity.rate_of_strain(physicalStrainRateTensor)).pow(2)
            ).pow((self.powerIndex - 1.0)/2.0)
        )

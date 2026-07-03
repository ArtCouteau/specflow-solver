from __future__ import annotations

import torch
from typing import Callable

from . import term
from .. import solution
from ...core import context
from ...core.numerics.operations import algebra, derivative, fourier
from ...core.physics import fields, variable


class LESModelTerm(term.Term):
    def __init__(self, tag: str, equation: str, variable: str, rank: variable.Rank):
        super().__init__(tag, equation, variable, rank)

        if not self.rank.name == 'VECTOR':
            raise NotImplementedError()

    def time_step_limiter(self) -> tuple[str, float, Callable[[solution.SolutionVariable, float], float]] | None:
        return (
            f'Diffusion/{self.equation}/{self.tag}',
            context.get().config.get(f'equations/{self.equation}/terms/{self.tag}/limiter', float, 0.5),
            self._diffusion_number
        )

    def _diffusion_number(self, variable: solution.SolutionVariable, dt: float) -> float:
        return (
            self._physical_eddy_viscosity(
                fields.physical.velocity.strain_rate_tensor(variable.get('velocity'))
            ).max().item()
            * dt
            / context.get().mesh.physical_grid_spacing()**2
        )

    def _get_viscosity(self, velocity: torch.Tensor) -> torch.Tensor:
        return fourier.to_spectral(
            self._physical_eddy_viscosity(
                fields.physical.velocity.strain_rate_tensor(velocity.data)
            )
        )

    def _physical_eddy_viscosity(self, physicalStrainRateTensor: torch.Tensor) -> torch.Tensor: ...

    def _physical_stress_tensor(self, physicalStrainRateTensor: torch.Tensor) -> torch.Tensor:
        return 2.0*algebra.mul_scalar_tensor(
            self._physical_eddy_viscosity(physicalStrainRateTensor),
            physicalStrainRateTensor
        )

    def diagnostic_fields(self, variable: solution.SolutionVariable, time: float) -> dict[str, torch.Tensor]:
        return {
            'eddyViscosity': fourier.to_spectral(
                self._physical_eddy_viscosity(
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


@term.TermRegistry.register('LES/Smagorinsky')
class SmagorinskyLESModelTerm(LESModelTerm):
    constant: float
    filterWidth: float

    def __init__(self, tag: str, equation: str, variable: str, rank: variable.Rank):
        super().__init__(tag, equation, variable, rank)

        self.constant = context.get().config.get(f'equations/{self.equation}/terms/{tag}/constant', float, 0.12)
        self.filterWidth = context.get().mesh.physical_grid_spacing()

    def _physical_viscosity(self, physicalStrainRateTensor: torch.Tensor) -> torch.Tensor:
        return (
            (self.constant*self.filterWidth)**2
            * fields.physical.velocity.rate_of_strain(physicalStrainRateTensor)
        )

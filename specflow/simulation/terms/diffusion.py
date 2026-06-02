from __future__ import annotations

import torch

from . import term
from ...core.numerics import context
from ...core.numerics.operations import derivative
from ...core.physics import variable


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

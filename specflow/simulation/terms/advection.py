from __future__ import annotations

import torch

from . import term
from .. import equation, solution
from ...core.numerics import context
from ...core.numerics.operations import algebra, derivative, fourier
from ...core.physics import fields, variable


class AdvectionTerm(term.Term):
    advectingField: str = 'velocity'
    advectedField: str

    def __init__(self, equation: equation.Equation, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)
        self.advectedField = self.equation.variable

        self.equation.solver.integrator.add_time_step_limiter(
            'CFL',
            context.get().config.get(f'equations/{equation.tag}/terms/{tag}/limiter', float, 0.5),
            self._CFL_number
        )

    def _CFL_number(self) -> float:
        return (
            fourier.to_physical(self.equation.solver.variable.get(self.advectingField)).abs().max().item()
            * self.equation.solver.integrator.dt
            / context.get().mesh.physical_grid_spacing()
        )


@term.TermRegistry.register('advection', 'rotation')
class RotationFormAdvectionTerm(AdvectionTerm):
    def __init__(self, equation: equation.Equation, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        if not self.rank.name == 'VECTOR':
            raise ValueError('Rotation form convection term can only be used in vector equations')
        if not context.get().config.get(f'grid/dimension', int) == 3:
            raise ValueError('Rotation form convection term can only be used in 3D simulations')

    def _physical_space_operation(self, velocity: torch.Tensor, vorticity: torch.Tensor) -> torch.Tensor:
        return - algebra.cross_vector_vector(vorticity, velocity)

    def compute(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor:
        return context.get().physics.dealias.apply(
            self._physical_space_operation,
            variable.get(self.advectingField),
            fields.spectral.velocity.vorticity(variable.get(self.advectedField))
        )


@term.TermRegistry.register('advection', 'convection')
class ConvectionFormAdvectionTerm(AdvectionTerm):
    gradient: derivative.Gradient
    operation: algebra.InterRankOperation

    def __init__(self, equation: equation.Equation, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        if self.rank.name == 'SCALAR':
            self.gradient = derivative.gradient_scalar
            self.operation = algebra.inner_vector_vector
        elif self.rank.name == 'VECTOR':
            self.gradient = derivative.gradient_vector
            self.operation = algebra.inner_vector_tensor
        else:
            raise NotImplementedError()

    def _physical_space_operation(self, velocity: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        return self.operation(velocity, gradient)

    def compute(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor:
        return context.get().physics.dealias.apply(
            self._physical_space_operation,
            variable.get(self.advectingField),
            self.gradient(variable.get(self.advectedField))
        )

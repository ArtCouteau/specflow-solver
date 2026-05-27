from __future__ import annotations

import torch

from . import term
from .. import solver
from ...core.numerics import context
from ...core.numerics.operations import algebra, derivative, fourier
from ...core.physics import fields, variable


class AdvectionTerm(term.Term):
    velocity: variable.Variable

    def __init__(self, equation: str, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        context.get().sim.integrator.add_time_step_limiter(
            tag,
            context.get().config.get(f'equations/{equation}/terms/{tag}/limiter', float, 0.5),
            self._CFL_number
        )

    def _CFL_number(self) -> float:
        return (
            fourier.to_physical(self.velocity.data).abs().max().item()
            * context.get().sim.integrator.dt
            / context.get().mesh.physical_grid_spacing()
        )

    def add_references(self, solver: solver.Solver) -> None:
        # TODO: hardcoded
        self.velocity = solver.equations['Navier-Stokes'].variable


@term.TermRegistry.register('advection', 'rotation')
class RotationFormAdvectionTerm(AdvectionTerm):
    def __init__(self, equation: str, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)

        if not self.rank.name == 'VECTOR':
            raise ValueError('Rotation form convection term can only be used in vector equations')
        if not context.get().config.get(f'grid/dimension', int) == 3:
            raise ValueError('Rotation form convection term can only be used in 3D simulations')

    def _physical_space_operation(self, velocity: torch.Tensor, vorticity: torch.Tensor) -> torch.Tensor:
        return - algebra.cross_vector_vector(vorticity, velocity)

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor:
        return context.get().physics.dealias.apply(
            self._physical_space_operation,
            variable,
            fields.spectral.velocity.vorticity(variable)
        )


@term.TermRegistry.register('advection', 'convection')
class ConvectionFormAdvectionTerm(AdvectionTerm):
    gradient: derivative.Gradient
    operation: algebra.InterRankOperation

    def __init__(self, equation: str, tag: str, rank: variable.Rank):
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

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor:
        return context.get().physics.dealias.apply(
            self._physical_space_operation,
            variable,
            self.gradient(variable)
        )

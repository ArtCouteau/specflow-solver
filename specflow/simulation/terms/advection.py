from __future__ import annotations

import torch

from ...core.numerics.context import get_context
from ...core.numerics.operations import algebra, derivative
from ...core.physics import fields
from . import term


@term.TermRegistry.register('advection', 'rotation')
class RotationFormAdvectionTerm(term.Term):
    def __init__(self, equation: str, tag: str):
        super().__init__(equation, tag)

        if not self.rank.name == 'VECTOR':
            raise ValueError('Rotation form convection term can only be used in vector equations')
        if not get_context().config.get(f'grid/dimension', int) == 3:
            raise ValueError('Rotation form convection term can only be used in 3D simulations')

    def _physical_space_operation(self, velocity: torch.Tensor, vorticity: torch.Tensor) -> torch.Tensor:
        return - algebra.cross_vector_vector(vorticity, velocity)

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor:
        return get_context().physics.dealias.apply(
            self._physical_space_operation,
            variable,
            fields.spectral.velocity.vorticity(variable)
        )


@term.TermRegistry.register('advection', 'convection')
class ConvectionFormAdvectionTerm(term.Term):
    gradient: derivative.Gradient
    operation: algebra.InterRankOperation

    def __init__(self, equation: str, tag: str):
        super().__init__(equation, tag)

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
        return get_context().physics.dealias.apply(
            self._physical_space_operation,
            variable,
            self.gradient(variable)
        )

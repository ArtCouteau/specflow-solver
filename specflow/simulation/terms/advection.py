from __future__ import annotations

import torch

from specflow.core.numerics.context import get_context
from specflow.core.numerics.operations import algebra
from specflow.simulation.terms import term


class RotationFormAdvectionTerm(term.Term):
    def __init__(self):
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
            # add vorticity
        )

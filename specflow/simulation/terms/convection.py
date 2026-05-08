from __future__ import annotations

import torch

from specflow.core.numerics.context import get_context
from specflow.simulation.terms import term


class ConvectionTerm(term.Term):
    pass


class RotationFormConvectionTerm(ConvectionTerm):
    def __init__(self):
        if not self.rank.name == 'VECTOR':
            raise ValueError('Rotation form convection term can only be used in vector equations')
        if not get_context().config.get(f'grid/dimension', int) == 3:
            raise ValueError('Rotation form convection term can only be used in 3D simulations')

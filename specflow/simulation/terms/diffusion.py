from __future__ import annotations

import torch

from ...core.numerics import context
from ...core.numerics.operations import derivative
from ...core.physics import variable
from . import term


@term.TermRegistry.register('diffusion', 'constant')
class ConstantDiffusivityDiffusionTerm(term.Term):
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

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor:
        return self.diffusivity * self.laplacian(variable)

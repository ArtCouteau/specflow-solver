from __future__ import annotations

import torch

from ...core.numerics.context import get_context
from ...core.numerics.operations import derivative
from . import term


class ConstantDiffusivityDiffusionTerm(term.Term):
    diffusivity: float
    laplacian: derivative.Laplacian

    def __init__(self, equation: str, tag: str):
        super().__init__(equation, tag)

        if self.rank.name == 'SCALAR':
            self.laplacian = derivative.laplacian_scalar
        elif self.rank.name == 'VECTOR':
            self.laplacian = derivative.laplacian_vector
        else:
            raise NotImplementedError()

        self.diffusivity = get_context().config.get(f'equations/{equation}/terms/{tag}/diffusivity', float)

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor:
        return self.diffusivity * self.laplacian(variable)

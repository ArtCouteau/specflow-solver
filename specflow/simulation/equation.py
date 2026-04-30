from __future__ import annotations

import torch

from .terms import term
from ..core.physics import field


class Equation:
    rank: field.Rank
    terms: dict[str, term.Term]
    variable: field.Field

    def __init__(self):
        pass

    def compute_rhs(self, variable: torch.Tensor, time: float):
        tmp = torch.zeros_like(variable)
        for term in self.terms.values():
            tmp.add_(term.compute(variable, time))

        return tmp

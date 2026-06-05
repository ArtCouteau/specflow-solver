from __future__ import annotations

import torch

from .. import equation, solution
from ...core.physics import variable
from ...core.registry import Registry


class Term:
    equation: equation.Equation
    tag: str
    rank: variable.Rank

    def __init__(self, equation: equation.Equation, tag: str, rank: variable.Rank):
        self.equation = equation
        self.tag = tag
        self.rank = rank

    def compute(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor: ...


class TermRegistry(Registry[Term]):
    pass

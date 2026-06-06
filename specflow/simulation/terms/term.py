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

    def reconstruct(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor:
        return self.compute(variable, time)

    def restart_state(self) -> dict[str, torch.Tensor]:
        return {}

    def diagnostic_fields(self, variable: solution.SolutionVariable, time: float) -> dict[str, torch.Tensor]:
        return {}


class TermRegistry(Registry[Term]):
    pass

from __future__ import annotations

import torch
from typing import Callable

from .. import solution
from ...core.physics import variable
from ...core.registry import Registry


class Term:
    tag: str
    equation: str
    variable: str
    rank: variable.Rank

    def __init__(self, tag: str, equation: str, variable: str, rank: variable.Rank):
        self.tag = tag
        self.equation = equation
        self.variable = variable
        self.rank = rank

    def compute(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor: ...

    def reconstruct(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor:
        return self.compute(variable, time)

    def time_step_limiter(self) -> tuple[str, float, Callable[[solution.SolutionVariable, float], float]] | None:
        return None

    def restart_state(self) -> dict[str, torch.Tensor]:
        return {}

    def diagnostic_fields(self, variable: solution.SolutionVariable, time: float) -> dict[str, torch.Tensor]:
        return {}


class TermRegistry(Registry[Term]):
    pass

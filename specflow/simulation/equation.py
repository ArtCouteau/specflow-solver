from __future__ import annotations

import torch

from . import solution, solver
from .terms import term
from ..core.numerics import context
from ..core.numerics.operations import projection, rank
from ..core.physics import variable


class Equation:
    projector: projection.Projector
    rank: variable.Rank
    tag: str
    terms: dict[str, term.Term]
    solver: solver.Solver
    variable: str

    def __init__(self, solver: solver.Solver, tag: str):
        self.solver = solver
        self.tag = tag
        self.rank = variable.Rank[context.get().config.get(f'equations/{self.tag}/rank', str).upper()]
        self.variable = context.get().config.get(f'equations/{self.tag}/variable', str)
        self.terms = {
            termTag: term.TermRegistry.get(
                f"{context.get().config.get(f'equations/{tag}/terms/{termTag}/kind', str)}"
                f"/{context.get().config.get(f'equations/{tag}/terms/{termTag}/model', str)}"
            )(self, termTag, self.rank)
            for termTag in context.get().config.get(f'equations/{tag}/terms', dict).keys()
        }
        self.projector = projection.ProjectorRegistry.get(
            context.get().config.get(f'equations/{self.tag}/projector', str)
        )()

    def compute_rhs(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor:
        tmp = rank.zeros(self.rank)
        for term in self.terms.values():
            tmp.add_(term.compute(variable, time))

        return self.projector.apply(tmp)

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

    def __init__(self, solver: solver.Solver, tag: str):
        self.solver = solver
        self.tag = tag
        self.rank = variable.Rank[context.get().config.get(f'equations/{self.tag}/rank', str).upper()]
        self.terms = {
            termTag: term.TermRegistry.get(
                context.get().config.get(f'equations/{tag}/terms/{termTag}/kind', str),
                context.get().config.get(f'equations/{tag}/terms/{termTag}/model', str)
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

    def write_state(self) -> None:
        if context.get().config.get('save/state', bool, True):
            context.get().io.h5.write_tensor(
                f'output/{self.solver.control.time.timestamp()}/simulation',
                f'{self.tag}/fields/{self.variable.tag}',
                self.variable.data,
                {'rank': self.rank.name}
            )

        # if context.get().config.get('save/terms', bool, False):
        #     for key, term in self.terms.items():
        #         context.get().io.h5.write_tensor(
        #             f'output/{context.get().sim.control.time.timestamp()}/simulation',
        #             f'{self.tag}/fields/terms/{key}',
        #             term.compute(self.variable.data, context.get().sim.control.time.current),
        #             {'rank': self.rank.name}
        #         )

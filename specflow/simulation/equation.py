from __future__ import annotations

import torch
from typing import Literal

from . import solution, solver
from .terms import term
from ..core.numerics import context
from ..core.numerics.operations import fourier, projection, rank
from ..core.physics import variable


class Equation:
    projector: projection.Projector
    rank: variable.Rank
    tag: str
    terms: dict[str, term.Term]
    variable: variable.Variable
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
        self._init_variable()

    def _init_variable(self) -> None:
        self.variable = variable.Variable(
            tag=context.get().config.get(f'equations/{self.tag}/variable', str),
            rank=self.rank,
            data=context.get().memory.alloc(
                f'{self.tag}/variable',
                variable.field_shape(self.rank),
                dtype=context.get().config.get('numerics/precision/complex', torch.dtype),
                device=context.get().config.get('numerics/device', str)
            )
        )

        self.variable.data.zero_()

    def set_initial_condition(self, data: torch.Tensor, space: Literal['physical', 'spectral']):
        if space == 'physical':
            self.variable.data.copy_(fourier.to_spectral(data))
        else:
            self.variable.data.copy_(data)

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

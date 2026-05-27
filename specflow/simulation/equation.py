from __future__ import annotations

import torch
from typing import Literal

from . import solver
from .terms import term
from ..core.numerics import context
from ..core.numerics.operations import fourier, projection
from ..core.physics import variable


class Equation:
    tag: str
    rank: variable.Rank
    terms: dict[str, term.Term]
    variable: variable.Variable

    def __init__(self, tag: str):
        self.tag = tag
        self.rank = variable.Rank[context.get().config.get(f'equations/{self.tag}/rank', str).upper()]
        self.terms = {
            termTag: term.TermRegistry.get(
                context.get().config.get(f'equations/{tag}/terms/{termTag}/kind', str),
                context.get().config.get(f'equations/{tag}/terms/{termTag}/model', str)
            )(self.tag, termTag, self.rank)
            for termTag in context.get().config.get(f'equations/{tag}/terms', dict).keys()
        }
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

    def add_references(self, solver: solver.Solver) -> None:
        for term in self.terms.values():
            term.add_references(solver)

    def set_initial_condition(self, data: torch.Tensor, space: Literal['physical', 'spectral']):
        if space == 'physical':
            self.variable.data.copy_(fourier.to_spectral(data))
        else:
            self.variable.data.copy_(data)

    def compute_rhs(self, variable: torch.Tensor, time: float) -> torch.Tensor:
        tmp = torch.zeros_like(variable)
        for term in self.terms.values():
            tmp.add_(term.compute(variable, time))

        # TODO: hardcoded
        return projection.to_divergence_free_plane(tmp)

    def step(self) -> None:
        _ctx = context.get()
        _ctx.sim.integrator.advance(self)

        if self.variable.data.isnan().any():
            raise ValueError(f'NaN values detected in the variable of equation {self.tag}')

    def write_state(self) -> None:
        if context.get().config.get('save/state', bool, True):
            context.get().io.h5.write_tensor(
                f'output/{context.get().sim.control.time.timestamp()}/simulation',
                f'{self.tag}/variable',
                self.variable.data,
                {'rank': self.rank.name}
            )

        if context.get().config.get('save/terms', bool, False):
            for key, term in self.terms.items():
                context.get().io.h5.write_tensor(
                    f'output/{context.get().sim.control.time.timestamp()}/simulation',
                    f'{self.tag}/terms/{key}',
                    term.compute(self.variable.data, context.get().sim.control.time.current),
                    {'rank': self.rank.name}
                )

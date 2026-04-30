from __future__ import annotations

import torch

from .terms import term
from ..core.numerics import get_context
from ..core.physics import field


class Equation:
    tag: str
    rank: field.Rank
    terms: dict[str, term.Term]
    variable: field.Field

    def __init__(self, tag: str):
        self.tag = tag
        self.rank = field.Rank[get_context().config.get(f'equations/{self.tag}/rank', str).upper()]
        self._init_variable()

    def _init_variable(self) -> None:
        self.variable = field.Field(
            tag=get_context().config.get(f'equations/{self.tag}/variable', str),
            rank=self.rank,
            data=get_context().memory.alloc(
                f'{self.tag}/variable',
                field.field_shape(self.rank),
                dtype=get_context().config.get('numerics/precision/complex', torch.dtype),
                device=get_context().config.get('numerics/device', str)
            )
        )

        self.variable.data.zero_()

    def compute_rhs(self, variable: torch.Tensor, time: float) -> torch.Tensor:
        tmp = torch.zeros_like(variable)
        for term in self.terms.values():
            tmp.add_(term.compute(variable, time))

        return tmp

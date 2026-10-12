from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar, override

import torch

from specflow.core.numerics.operations import fourier
from specflow.core.numerics.operations.projection import Projector
from specflow.core.physics.wrappers.constant import Constant
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, Scalar, Spectral, TRank, Vector
from specflow.processing.numerics.node import Node
from specflow.processing.numerics.rule import Rule
from specflow.simulation.terms.term import Term

TRankEquation = TypeVar("TRankEquation", bound=Scalar | Vector)

TIME: Node[Constant[Scalar]] = Node("TimeConstant", Constant[Scalar])


@dataclass(frozen=True)
class Equation(Generic[TRank]):
    tag: str
    rank: type[TRank]
    terms: dict[str, Term[TRank]]
    rhs: Field[TRank, Spectral]
    variable: Field[TRank, Spectral]

    def compute_rhs(self, time: float) -> Field[TRank, Spectral]:
        self.rhs.zero_()
        for term in self.terms.values():
            term.compute(self.rhs, time)

        return self.rhs

    def rules(self) -> list[Rule[Any]]:
        rules: list[Rule[Any]] = []
        for term in self.terms.values():
            contribution = Node(f"{self.tag}{term.tag}Contribution", self.variable.kind)
            rules.append(Rule(contribution, partial(self._contribution, term), (TIME,)))
            rules.extend(term.rules())
        return rules

    def _contribution(self, term: Term[TRank], time: Constant[Scalar]) -> Field[TRank, Spectral]:
        out = self.variable.replace(data=torch.zeros_like(self.variable.data))
        term.contribution(out, time.value)
        return out

    def set_initial_conditions_physical(self, field: Field[TRank, Physical]) -> None:
        self.variable.copy_(fourier.to_spectral(field))

    def set_initial_conditions_spectral(self, field: Field[TRank, Spectral]) -> None:
        self.variable.copy_(field)


@dataclass(frozen=True)
class NavierStokesEquation(Equation[Vector]):
    projector: Projector

    @override
    def compute_rhs(self, time: float) -> Field[Vector, Spectral]:
        return self.rhs.copy_(self.projector.apply(self.variable.mesh, super().compute_rhs(time)))

    @override
    def _contribution(self, term: Term[Vector], time: Constant[Scalar]) -> Field[Vector, Spectral]:
        return self.projector.apply(self.variable.mesh, super()._contribution(term, time))

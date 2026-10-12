from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic

from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Spectral, TRank
from specflow.core.registry import Registry
from specflow.processing.numerics.rule import Rule


@dataclass(frozen=True)
class Term(Generic[TRank]):
    rank: type[TRank]
    tag: str

    @abstractmethod
    def compute(self, rhs: Field[TRank, Spectral], time: float) -> None: ...

    def time_step_limiter(self) -> tuple[str, float, Callable[[Field[TRank, Spectral], float], float]] | None:
        return None

    def diagnostic_fields(self, time: float) -> dict[str, Field[TRank, Spectral]]:
        return {}

    def contribution(self, out: Field[TRank, Spectral], time: float) -> None:
        # Adds the term to out for post-processing, must leave the term state untouched
        self.compute(out, time)

    def rules(self) -> list[Rule[Any]]:
        return []


class TermRegistry(Registry[Term[TRank]]):
    pass

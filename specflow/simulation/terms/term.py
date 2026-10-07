from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Spectral, TRank
from specflow.core.registry import Registry


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


class TermRegistry(Registry[Term[TRank]]):
    pass

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from specflow.core.numerics.backend.backend import NumericsBackend
from specflow.core.numerics.mesh import Mesh
from specflow.core.physics.wrappers.wrapper import Wrapper
from specflow.processing.physics.statistics.quantity import Quantity

TValue = TypeVar("TValue", bound=Wrapper)


@dataclass(frozen=True)
class Frame:
    _quantities: dict[type[Quantity[Any]], Any] = field(init=False)
    backend: NumericsBackend
    mesh: Mesh

    def __getitem__(self, quantity: type[Quantity[TValue]]) -> TValue:
        if quantity not in self._quantities:
            raise KeyError(f"{quantity.__name__} not in frame, has: {[q.__name__ for q in self._quantities]}")
        return self._quantities[quantity]

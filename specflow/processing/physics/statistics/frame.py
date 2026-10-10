from __future__ import annotations

from typing import Any, TypeVar

from specflow.core.physics.wrappers.wrapper import Wrapper
from specflow.processing.physics.statistics.quantity import Quantity

TValue = TypeVar("TValue", bound=Wrapper)


class Frame:
    _quantities: dict[type[Quantity[Any]], Quantity[Any]]

    def __getitem__(self, quantity: type[Quantity[TValue]]) -> Quantity[TValue]:
        if quantity not in self._quantities:
            raise KeyError(f"{quantity.__name__} not in frame, has: {[q.__name__ for q in self._quantities]}")
        return self._quantities[quantity]

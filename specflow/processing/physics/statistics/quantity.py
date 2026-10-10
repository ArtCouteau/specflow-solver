from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

from specflow.core.physics.wrappers.wrapper import Wrapper
from specflow.processing.physics.statistics.frame import Frame

TValue = TypeVar("TValue", bound=Wrapper)


@dataclass(frozen=True)
class Quantity(Generic[TValue]):
    value: TValue


class Input(Quantity[TValue]): ...


class Derived(Quantity[TValue]):
    @abstractmethod
    def compute(self, frame: Frame) -> Self: ...

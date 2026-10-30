from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Rank(IntEnum):
    SCALAR = 0
    VECTOR = 1
    TENSOR = 2


@dataclass(frozen=True)
class Quantity:
    name: str
    rank: Rank


class Space(Enum):
    PHYSICAL = 0
    SPECTRAL = 1


@dataclass(frozen=True)
class Field(Quantity):
    space: Space

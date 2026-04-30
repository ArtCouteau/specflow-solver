from __future__ import annotations

import enum
import torch


class Rank(enum.Enum):
    SCALAR = 0
    VECTOR = 1
    TENSOR = 2


class Field:
    tag: str
    rank: Rank
    data: torch.Tensor

    def __init__(self):
        pass

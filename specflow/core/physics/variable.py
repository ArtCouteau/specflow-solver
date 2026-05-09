from __future__ import annotations

import enum
import torch

from ..numerics import get_context


class Rank(enum.Enum):
    SCALAR = 0
    VECTOR = 1
    TENSOR = 2


class Variable:
    tag: str
    rank: Rank
    data: torch.Tensor

    def __init__(self, tag: str, rank: Rank, data: torch.Tensor):
        self.tag = tag
        self.rank = rank
        self.data = data


def field_shape(rank: Rank) -> tuple[int, ...]:
    DIM: int = get_context().config.get('grid/dimension', int)

    if rank == Rank.SCALAR:
        return get_context().mesh.domain_shape()
    elif rank == Rank.VECTOR:
        return (DIM,) + get_context().mesh.domain_shape()
    elif rank == Rank.TENSOR:
        return (DIM, DIM,) + get_context().mesh.domain_shape()

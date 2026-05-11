from __future__ import annotations

import enum
import torch

from ..numerics import context


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
    DIM: int = context.get().config.get('grid/dimension', int)

    if rank == Rank.SCALAR:
        return context.get().mesh.domain_shape()
    elif rank == Rank.VECTOR:
        return (DIM,) + context.get().mesh.domain_shape()
    elif rank == Rank.TENSOR:
        return (DIM, DIM,) + context.get().mesh.domain_shape()

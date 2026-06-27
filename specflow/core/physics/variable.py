from __future__ import annotations

import enum
import torch

from ..numerics import context
from ..numerics.decomposition import communication


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


def _domain_shape() -> tuple[int, ...]:
    N: int = context.get().config.get('grid/N', int)
    DIM: int = context.get().config.get('grid/dimension', int)

    shape = (N,) * (DIM - 1) + (N // 2 + 1,)

    if context.get().distribution.enabled:
        assert DIM == 3, 'Distributed mode currently only supports 3D problems (grid/dimension == 3)'
        shape = (communication.local_size(N),) + shape[1:]

    return shape


def field_shape(rank: Rank) -> tuple[int, ...]:
    DIM: int = context.get().config.get('grid/dimension', int)

    if rank == Rank.SCALAR:
        return _domain_shape()
    elif rank == Rank.VECTOR:
        return (DIM,) + _domain_shape()
    elif rank == Rank.TENSOR:
        return (DIM, DIM,) + _domain_shape()

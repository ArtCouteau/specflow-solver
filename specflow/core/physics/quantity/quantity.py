from dataclasses import dataclass
from typing import TypeVar

import torch


class Rank: ...


class Scalar(Rank): ...


class Vector(Rank): ...


class Tensor(Rank): ...


class Space: ...


class Physical(Space): ...


class Spectral(Space): ...


TRank = TypeVar("TRank", bound=Rank)
TSpace = TypeVar("TSpace", bound=Space)


@dataclass(frozen=True, eq=False)
class Quantity:
    data: torch.Tensor

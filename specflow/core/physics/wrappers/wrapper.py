from dataclasses import dataclass
from typing import Any, TypeVar

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
class Wrapper:
    data: torch.Tensor

    @property
    def kind(self) -> Any:
        return type(self)

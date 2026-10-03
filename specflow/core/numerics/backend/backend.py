from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from specflow.core.numerics.backend.io import IOInterface
from specflow.core.numerics.backend.memory import MemoryManager
from specflow.core.numerics.backend.reduction import ReductionBackend
from specflow.core.numerics.backend.transforms import TransformsBackend


@dataclass(frozen=True)
class NumericsBackend:
    io: IOInterface
    memory: MemoryManager
    parameters: NumericsParameters
    reduction: ReductionBackend
    transforms: TransformsBackend


@dataclass(frozen=True)
class NumericsParameters:
    device: Literal["cpu", "cuda"]
    precision: Precision


@dataclass(frozen=True)
class Precision:
    INT: torch.dtype
    FLOAT: torch.dtype
    COMPLEX: torch.dtype

    @staticmethod
    def from_precision(precision: Literal["single", "double"]) -> Precision:
        if precision == "single":
            return Precision(torch.int64, torch.float32, torch.complex64)
        elif precision == "double":
            return Precision(torch.int64, torch.float64, torch.complex128)

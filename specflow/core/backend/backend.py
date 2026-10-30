from __future__ import annotations

from dataclasses import dataclass

from specflow.core.backend.reduction import Reduction
from specflow.core.backend.transform import Transform


@dataclass(frozen=True)
class Backend:
    reduction: Reduction
    transform: Transform

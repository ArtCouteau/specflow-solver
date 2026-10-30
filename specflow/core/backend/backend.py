from __future__ import annotations

from dataclasses import dataclass

from matplotlib.transforms import Transform


@dataclass(frozen=True)
class Backend:
    transform: Transform

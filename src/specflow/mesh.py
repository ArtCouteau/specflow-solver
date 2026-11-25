"""The mesh of the periodic domain."""

from dataclasses import dataclass

from specflow.decomposition import Decomposition


@dataclass(frozen=True, eq=False)
class Mesh:
    """N equispaced points per direction of the periodic domain [0, L)^d."""

    points: int
    length: float
    dimension: int
    decomposition: Decomposition

    def __post_init__(self) -> None:
        if self.points < 2 or self.points % 2:
            raise ValueError(f"points must be even and at least 2, got {self.points}")

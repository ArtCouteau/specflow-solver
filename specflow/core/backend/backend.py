from __future__ import annotations

from dataclasses import dataclass

from specflow.core.backend.reduction import Reduction, SerialReduction
from specflow.core.backend.transform import SerialTransform, Transform
from specflow.core.numerics.mesh import Mesh


@dataclass(frozen=True)
class Backend:
    reduction: Reduction
    transform: Transform


def get_serial(mesh: Mesh) -> Backend:
    return Backend(SerialReduction(), SerialTransform(mesh))

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Generic, Literal

import torch

from specflow.core.numerics.backend.backend import NumericsBackend
from specflow.core.numerics.mesh import Mesh
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, Scalar, TRank, Vector, Wrapper


@dataclass(frozen=True)
class Constant(Wrapper, Generic[TRank]):
    rank: type[TRank]
    component: Literal[1, 2, 3]
    dimension: Literal[1, 2, 3]
    backend: NumericsBackend = field(repr=False)
    mesh: Mesh = field(repr=False)

    @staticmethod
    def scalar(value: float, backend: NumericsBackend, mesh: Mesh) -> Constant[Scalar]:
        return Constant(
            torch.tensor(value, dtype=backend.parameters.precision.FLOAT),
            Scalar,
            1,
            mesh.layout.DIMENSION,
            backend,
            mesh,
        )

    @staticmethod
    def vector(*values: float, backend: NumericsBackend, mesh: Mesh) -> Constant[Vector]:
        if not len(values) == mesh.layout.DIMENSION:
            raise ValueError(
                f"Number of values provided ({len(values)}) does not match mesh dimension ({mesh.layout.DIMENSION})"
            )
        return Constant(
            torch.tensor(values, dtype=backend.parameters.precision.FLOAT),
            Vector,
            mesh.layout.DIMENSION,
            mesh.layout.DIMENSION,
            backend,
            mesh,
        )

    @cached_property
    def value(self: Constant[Scalar]) -> float:
        return float(self.data)

    def broadcast_like(self, field: Field[TRank, Physical]) -> Field[TRank, Physical]:
        return field.replace(
            data=self.data.to(dtype=field.data.dtype, device=field.data.device)
            .view(*self.data.shape, *(1,) * field.dimension)
            .expand_as(field.data)
        )

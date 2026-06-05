from __future__ import annotations

import torch

from . import algebra, derivative, rank
from .. import context
from ...registry import Registry


class Projector:
    def apply(self, vector: torch.Tensor) -> torch.Tensor: ...


class ProjectorRegistry(Registry[Projector]):
    pass


@ProjectorRegistry.register('none')
class NoneProjector(Projector):
    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        return vector


@ProjectorRegistry.register('divergenceFree')
class DivergenceFreeProjector(Projector):
    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        pressureCorrection = algebra.outer_vector_vector(
            context.get().mesh.grid,
            derivative.inv_nabla2()*context.get().mesh.grid
        )

        return algebra.matmul_tensor_vector(
            rank.kronecker_delta_like(pressureCorrection) + pressureCorrection,
            vector
        )

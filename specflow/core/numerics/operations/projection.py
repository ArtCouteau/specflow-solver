from __future__ import annotations

import torch

from . import algebra, derivative
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
        k = context.get().mesh.k

        return vector - k * algebra.inner_vector_vector(k, vector) * (- derivative.inv_nabla2())

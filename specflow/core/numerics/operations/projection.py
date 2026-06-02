from __future__ import annotations

import torch
from typing import Type

from . import algebra, derivative, rank
from .. import context


class ProjectorRegistry:
    _registry: dict[str, Type[Projector]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(dealiasing: Type[Projector]) -> Type[Projector]:
            cls._registry[name] = dealiasing
            return dealiasing
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[Projector]:
        if not name in cls._registry:
            raise KeyError(
                f'Projector \'{name}\' not found, available:\n'
                + '\n'.join(f"- {k}" for k in cls._registry.keys())
            )
        return cls._registry[name]


class Projector:
    def apply(self, vector: torch.Tensor) -> torch.Tensor: ...


@ProjectorRegistry.register('None')
class NoneProjector(Projector):
    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        return vector


@ProjectorRegistry.register('DivergenceFree')
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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

from specflow.core.numerics.mesh import Mesh
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Spectral, Vector

from ...registry import Registry
from . import algebra


class Projector(ABC):
    @abstractmethod
    def apply(self, mesh: Mesh, vector: Field[Vector, Spectral]) -> Field[Vector, Spectral]: ...


class ProjectorRegistry(Registry[Projector]): ...


@ProjectorRegistry.register("divergenceFree")
class DivergenceFreeProjector(Projector):
    @override
    def apply(self, mesh: Mesh, vector: Field[Vector, Spectral]) -> Field[Vector, Spectral]:
        return vector - algebra.mul_scalar_vector(
            algebra.mul_scalar_scalar(
                algebra.inner_vector_vector(mesh.spectral.wavenumbers, vector),
                mesh.spectral.inverseWavenumbersSquared,
            ),
            mesh.spectral.wavenumbers,
        )

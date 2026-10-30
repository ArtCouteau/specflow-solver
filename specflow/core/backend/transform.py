from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp
from jax import Array
from jaxtyping import Complex, Float

from specflow.core.numerics.mesh import Mesh


class Transform(Protocol):
    def to_physical(self, field: Complex[Array, "*lead N//2+1"]) -> Float[Array, "*lead N"]: ...
    def to_spectral(self, field: Float[Array, "*lead N"]) -> Complex[Array, "*lead N//2+1"]: ...
    def to_physical_complex(self, field: Complex[Array, "*lead N"]) -> Float[Array, "*lead N"]: ...
    def to_spectral_complex(self, field: Float[Array, "*lead N"]) -> Complex[Array, "*lead N"]: ...


@dataclass(frozen=True)
class SerialTransform:
    mesh: Mesh

    def to_physical(self, field: Complex[Array, "*lead N//2+1"]) -> Float[Array, "*lead N"]:
        return jnp.fft.irfftn(field, axes=self.mesh.domain_axes, norm="forward")

    def to_spectral(self, field: Float[Array, "*lead N"]) -> Complex[Array, "*lead N//2+1"]:
        return jnp.fft.rfftn(field, axes=self.mesh.domain_axes, norm="forward")

    def to_physical_complex(self, field: Complex[Array, "*lead N"]) -> Float[Array, "*lead N"]:
        return jnp.fft.ifftn(field, axes=self.mesh.domain_axes, norm="forward")

    def to_spectral_complex(self, field: Float[Array, "*lead N"]) -> Complex[Array, "*lead N"]:
        return jnp.fft.fftn(field, axes=self.mesh.domain_axes, norm="forward")

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array
from jaxtyping import Int, Float, Complex

from specflow.core.numerics.operations import algebra


@dataclass(frozen=True)
class Mesh:
    number_of_points: int
    length: float
    dimension: Literal[1, 2, 3]

    @property
    def domain_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.dimension, 0))

    @property
    def vector_axis(self) -> int:
        return -(self.dimension + 1)

    @property
    def tensor_axes(self) -> tuple[int, int]:
        return (-(self.dimension + 2), -(self.dimension + 1))

    def wavenumbers(self) -> Int[Array, "*grid"]:
        one_dimensional = [
            jnp.fft.fftfreq(self.number_of_points, 1.0 / self.number_of_points) for _ in range(self.dimension - 1)
        ]
        one_dimensional.append(jnp.fft.rfftfreq(self.number_of_points, 1.0 / self.number_of_points))

        return jnp.stack(jnp.meshgrid(*one_dimensional, indexing="ij"))

    def gradient_scalar(self, scalar: Complex[Array, "*grid"]) -> Complex[Array, " {self.dimension} *grid"]:
        return algebra.scale_field(scalar, self._nabla())

    def gradient_vector(
        self, vector: Complex[Array, " {self.dimension}*grid"]
    ) -> Complex[Array, " {self.dimension} {self.dimension} *grid"]:
        return algebra.outer_vector_vector(self._nabla(), vector, self)

    def _nabla(self) -> Complex[Array, " {self.dimension} *grid"]:
        return 1.0j * self.wavenumbers()

    def _nabla_squared(self) -> Float[Array, "*grid"]:
        return -1.0 * self.wavenumbers() ** 2

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array
from jaxtyping import Int


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

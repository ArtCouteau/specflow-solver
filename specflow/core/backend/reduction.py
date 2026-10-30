from dataclasses import dataclass
from typing import Protocol

from jax import Array
from jaxtyping import Float


class Reduction(Protocol):
    def max(self, field: Float[Array, "*grid"]) -> Float[Array, ""]: ...
    def min(self, field: Float[Array, "*grid"]) -> Float[Array, ""]: ...
    def mean(self, field: Float[Array, "*grid"]) -> Float[Array, ""]: ...


@dataclass(frozen=True)
class SerialReduction:
    def max(self, field: Float[Array, "*grid"]) -> Float[Array, ""]:
        return field.max()

    def min(self, field: Float[Array, "*grid"]) -> Float[Array, ""]:
        return field.min()

    def mean(self, field: Float[Array, "*grid"]) -> Float[Array, ""]:
        return field.mean()

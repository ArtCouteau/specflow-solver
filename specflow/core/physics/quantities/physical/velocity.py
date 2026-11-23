import jax.numpy as jnp
from jax import Array
from jaxtyping import Complex, Float

from specflow.core.backend.backend import Backend
from specflow.core.numerics.mesh import Mesh


def velocity_physical_field(
    velocity_spectral_field: Complex[Array, "*lead N//2+1"], backend: Backend
) -> Float[Array, "*lead N"]:
    return backend.transform.to_physical(velocity_spectral_field)


def vorticity_physical_field(
    vorticity_spectral_field: Complex[Array, "*lead N//2+1"], backend: Backend
) -> Float[Array, "*lead N"]:
    return backend.transform.to_physical(vorticity_spectral_field)

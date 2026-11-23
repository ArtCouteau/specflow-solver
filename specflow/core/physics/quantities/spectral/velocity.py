import jax.numpy as jnp
from jax import Array
from jaxtyping import Complex, Float

from specflow.core.backend.backend import Backend
from specflow.core.numerics.mesh import Mesh


def vorticity_spectral_field(
    velocity_spectral_field: Complex[Array, "C *N N//2+1"], mesh: Mesh
) -> Complex[Array, "C *N N//2+1"]:
    return mesh.curl_vector_3D(velocity_spectral_field)


def velocity_gradient_spectral_field(
    velocity_spectral_field: Complex[Array, "C *N N//2+1"], mesh: Mesh
) -> Complex[Array, "C C *N N//2+1"]:
    return mesh.gradient_vector(velocity_spectral_field)


def strain_rate_tensor_spectral_field(
    velocity_gradient_spectral_field: Complex[Array, "C C *N N//2+1"], mesh: Mesh
) -> Complex[Array, "C C *N N//2+1"]:
    return

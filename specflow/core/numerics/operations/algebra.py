import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from specflow.core.numerics.mesh import Mesh


def cross_vector_vector_3D(
    a: Float[Array, "3, *grid"], b: Float[Array, "3, *grid"], mesh: Mesh
) -> Float[Array, "3, *grid"]:
    return jnp.linalg.cross(a, b, axis=mesh.vector_axis)


def inner_vector_vector(
    a: Float[Array, "components *grid"], b: Float[Array, "components *grid"], mesh: Mesh
) -> Float[Array, "*grid"]:
    return jnp.sum(a * b, axis=mesh.vector_axis)


def inner_tensor_tensor(
    a: Float[Array, "components components *grid"], b: Float[Array, "components components *grid"], mesh: Mesh
) -> Float[Array, "*grid"]:
    return jnp.sum(a * b, axis=mesh.tensor_axes)


def inner_vector_tensor(
    vector: Float[Array, "components *grid"], tensor: Float[Array, "components components *grid"], mesh: Mesh
) -> Float[Array, "components *grid"]:
    """Contract the vector with the first tensor index: result_j = vector_i * tensor_ij."""
    row_axis, column_axis = mesh.tensor_axes
    return jnp.sum(jnp.expand_dims(vector, column_axis) * tensor, axis=row_axis)


def matmul_tensor_vector(
    tensor: Float[Array, "components components *grid"], vector: Float[Array, "components *grid"], mesh: Mesh
) -> Float[Array, "components *grid"]:
    """Contract the vector with the second tensor index: result_i = tensor_ij * vector_j."""
    row_axis, column_axis = mesh.tensor_axes
    return jnp.sum(tensor * jnp.expand_dims(vector, row_axis), axis=column_axis)


def outer_vector_vector(
    a: Float[Array, "components *grid"], b: Float[Array, "components *grid"], mesh: Mesh
) -> Float[Array, "components components *grid"]:
    row_axis, column_axis = mesh.tensor_axes
    return jnp.expand_dims(a, column_axis) * jnp.expand_dims(b, row_axis)


def scale_field(scalar: Float[Array, "*grid"], field: Float[Array, "..."]) -> Float[Array, "..."]:
    return scalar * field

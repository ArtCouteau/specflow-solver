from specflow.core.numerics.mesh import Mesh
from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Scalar, Spectral, Tensor, Vector

from . import algebra


def nabla(mesh: Mesh) -> Field[Vector, Spectral]:
    return 1.0j * mesh.spectral.wavenumbers


def nabla_square(mesh: Mesh) -> Field[Scalar, Spectral]:
    return -mesh.spectral.wavenumbersSquared


def curl_vector(mesh: Mesh, vector: Field[Vector, Spectral]) -> Field[Vector, Spectral]:
    return algebra.cross_vector_vector(nabla(mesh), vector)


def divergence_vector(mesh: Mesh, vector: Field[Vector, Spectral]) -> Field[Scalar, Spectral]:
    return algebra.inner_vector_vector(nabla(mesh), vector)


def divergence_tensor(mesh: Mesh, tensor: Field[Tensor, Spectral]) -> Field[Vector, Spectral]:
    return algebra.inner_vector_tensor(nabla(mesh), tensor)


def gradient_scalar(mesh: Mesh, scalar: Field[Scalar, Spectral]) -> Field[Vector, Spectral]:
    return algebra.mul_scalar_vector(scalar, nabla(mesh))


def gradient_vector(mesh: Mesh, vector: Field[Vector, Spectral]) -> Field[Tensor, Spectral]:
    return algebra.outer_vector_vector(nabla(mesh), vector)


def laplacian_scalar(mesh: Mesh, scalar: Field[Scalar, Spectral]) -> Field[Scalar, Spectral]:
    return algebra.mul_scalar_scalar(nabla_square(mesh), scalar)


def laplacian_vector(mesh: Mesh, vector: Field[Vector, Spectral]) -> Field[Vector, Spectral]:
    return algebra.mul_scalar_vector(nabla_square(mesh), vector)

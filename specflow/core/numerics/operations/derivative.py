from specflow.core.numerics.mesh import Mesh
from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Scalar, Spectral, Tensor, Vector

from . import algebra


def nabla(mesh: Mesh) -> Field[Vector, Spectral]:
    return 1.0j * mesh.spectral.wavenumbers


def nabla_square(mesh: Mesh) -> Field[Scalar, Spectral]:
    return -mesh.spectral.wavenumbersSquared


def curl_vector(vector: Field[Vector, Spectral]) -> Field[Vector, Spectral]:
    return algebra.cross_vector_vector(nabla(vector.mesh), vector)


def divergence_vector(vector: Field[Vector, Spectral]) -> Field[Scalar, Spectral]:
    return algebra.inner_vector_vector(nabla(vector.mesh), vector)


def divergence_tensor(tensor: Field[Tensor, Spectral]) -> Field[Vector, Spectral]:
    return algebra.inner_vector_tensor(nabla(tensor.mesh), tensor)


def gradient_scalar(scalar: Field[Scalar, Spectral]) -> Field[Vector, Spectral]:
    return algebra.mul_scalar_vector(scalar, nabla(scalar.mesh))


def gradient_vector(vector: Field[Vector, Spectral]) -> Field[Tensor, Spectral]:
    return algebra.outer_vector_vector(nabla(vector.mesh), vector)


def laplacian_scalar(scalar: Field[Scalar, Spectral]) -> Field[Scalar, Spectral]:
    return algebra.mul_scalar_scalar(nabla_square(scalar.mesh), scalar)


def laplacian_vector(vector: Field[Vector, Spectral]) -> Field[Vector, Spectral]:
    return algebra.mul_scalar_vector(nabla_square(vector.mesh), vector)

from specflow.core.numerics.operations import derivative, norm, rank
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Scalar, Spectral, Tensor, Vector


def vorticity(velocity: Field[Vector, Spectral]) -> Field[Vector, Spectral]:
    return derivative.curl_vector(velocity)


def strain_rate_tensor(velocity: Field[Vector, Spectral]) -> Field[Tensor, Spectral]:
    return 0.5 * (derivative.gradient_vector(velocity) + rank.transpose(derivative.gradient_vector(velocity)))


def rotation_rate_tensor(velocity: Field[Vector, Spectral]) -> Field[Tensor, Spectral]:
    return 0.5 * (derivative.gradient_vector(velocity) - rank.transpose(derivative.gradient_vector(velocity)))


def modal_kinetic_energy(velocity: Field[Vector, Spectral]) -> Field[Scalar, Spectral]:
    return 0.5 * norm.square_l2_norm(velocity)

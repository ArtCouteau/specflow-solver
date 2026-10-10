from specflow.core.numerics.operations import algebra, derivative, fourier, norm, rank
from specflow.core.physics import fields
from specflow.core.physics.wrappers.constant import Constant
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, Scalar, Spectral, Tensor, Vector


def physical_velocity(velocity: Field[Vector, Spectral]) -> Field[Vector, Physical]:
    return fourier.to_physical(velocity)


def velocity_gradient(velocity: Field[Vector, Spectral]) -> Field[Tensor, Physical]:
    return fourier.to_physical(derivative.gradient_vector(velocity))


def vorticity(velocity: Field[Vector, Spectral]) -> Field[Vector, Physical]:
    return fourier.to_physical(fields.spectral.velocity.vorticity(velocity))


def strain_rate_tensor(velocityGradient: Field[Tensor, Physical]) -> Field[Tensor, Physical]:
    return 0.5 * (velocityGradient + rank.transpose(velocityGradient))


def rotation_rate_tensor(velocityGradient: Field[Tensor, Physical]) -> Field[Tensor, Physical]:
    return 0.5 * (velocityGradient - rank.transpose(velocityGradient))


def kinetic_energy(velocity: Field[Vector, Physical]) -> Field[Scalar, Physical]:
    return 0.5 * norm.square_l2_norm(velocity)


def Reynolds_stress_tensor(velocity: Field[Vector, Physical]) -> Field[Tensor, Physical]:
    return algebra.outer_vector_vector(velocity, velocity)


def enstrophy(rotationRateTensor: Field[Tensor, Physical]) -> Field[Scalar, Physical]:
    # |omega|^2 / 2 = Omega:Omega
    return norm.square_frobenius_norm(rotationRateTensor)


def dissipation(strainRateTensor: Field[Tensor, Physical], viscosity: Constant[Scalar]) -> Field[Scalar, Physical]:
    return 2.0 * viscosity.value * norm.square_frobenius_norm(strainRateTensor)


def dissipation_from_velocity_gradient(
    velocityGradient: Field[Tensor, Physical], viscosity: Constant[Scalar]
) -> Field[Scalar, Physical]:
    return viscosity.value * norm.square_frobenius_norm(velocityGradient)


def dissipation_from_vorticity(
    vorticity: Field[Vector, Physical], viscosity: Constant[Scalar]
) -> Field[Scalar, Physical]:
    return viscosity.value * norm.square_l2_norm(vorticity)


def rate_of_strain(strainRateTensor: Field[Tensor, Physical]) -> Field[Scalar, Physical]:
    return (2.0 * norm.square_frobenius_norm(strainRateTensor)).sqrt_()

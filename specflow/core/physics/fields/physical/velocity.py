from specflow.core.numerics.operations import algebra, derivative, fourier, norm, rank
from specflow.core.physics import fields
from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Physical, Scalar, Spectral, Tensor, Vector


def physical_velocity(velocity: Field[Vector, Spectral]) -> Field[Vector, Physical]:
    return fourier.to_physical(velocity)


def vorticity(velocity: Field[Vector, Spectral]) -> Field[Vector, Physical]:
    return fourier.to_physical(fields.spectral.velocity.vorticity(velocity))


def strain_rate_tensor(velocity: Field[Vector, Spectral]) -> Field[Tensor, Physical]:
    velocityGradient = fourier.to_physical(derivative.gradient_vector(velocity))
    return 0.5 * (velocityGradient + rank.transpose(velocityGradient))


def kinetic_energy(velocity: Field[Vector, Spectral]) -> Field[Scalar, Physical]:
    return 0.5 * norm.square_l2_norm(physical_velocity(velocity))


def Reynolds_stress_tensor(velocity: Field[Vector, Spectral]) -> Field[Tensor, Physical]:
    return algebra.outer_vector_vector(physical_velocity(velocity), physical_velocity(velocity))


def enstrophy(velocity: Field[Vector, Spectral]) -> Field[Scalar, Physical]:
    return 0.5 * norm.square_l2_norm(vorticity(velocity))


def dissipation(velocity: Field[Vector, Spectral], viscosity: float) -> Field[Scalar, Physical]:
    return 2.0 * viscosity * norm.square_frobenius_norm(strain_rate_tensor(velocity))


def dissipation_from_velocity_gradient(velocity: Field[Vector, Spectral], viscosity: float) -> Field[Scalar, Physical]:
    return viscosity * norm.square_frobenius_norm(fourier.to_physical(derivative.gradient_vector(velocity)))


def dissipation_from_vorticity(velocity: Field[Vector, Spectral], viscosity: float) -> Field[Scalar, Physical]:
    return viscosity * norm.square_l2_norm(vorticity(velocity))


def rate_of_strain(strainRateTensor: Field[Tensor, Physical]) -> Field[Scalar, Physical]:
    return (2.0 * norm.square_frobenius_norm(strainRateTensor)).sqrt_()

from specflow.core.numerics.operations import algebra, fourier, norm
from specflow.core.physics import fields
from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Physical, Scalar, Spectral, Vector


def residual_dissipation(
    velocity: Field[Vector, Spectral], eddyViscosity: Field[Scalar, Spectral]
) -> Field[Scalar, Physical]:
    return 2.0 * algebra.mul_scalar_scalar(
        fourier.to_physical(eddyViscosity),
        norm.square_frobenius_norm(fields.physical.velocity.strain_rate_tensor(velocity)),
    )

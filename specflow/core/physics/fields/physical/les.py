from specflow.core.numerics.operations import algebra, fourier, norm
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, Scalar, Spectral, Tensor


def residual_dissipation(
    strainRateTensor: Field[Tensor, Physical], eddyViscosity: Field[Scalar, Spectral]
) -> Field[Scalar, Physical]:
    return 2.0 * algebra.mul_scalar_scalar(
        fourier.to_physical(eddyViscosity), norm.square_frobenius_norm(strainRateTensor)
    )

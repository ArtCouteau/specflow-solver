from specflow.core.numerics.operations import algebra, norm
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, Scalar, Tensor


def residual_dissipation(
    strainRateTensor: Field[Tensor, Physical], eddyViscosity: Field[Scalar, Physical]
) -> Field[Scalar, Physical]:
    return 2.0 * algebra.mul_scalar_scalar(eddyViscosity, norm.square_frobenius_norm(strainRateTensor))

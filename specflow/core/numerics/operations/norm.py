from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Scalar, Tensor, TSpace, Vector


def l2_norm(field: Field[Vector, TSpace]) -> Field[Scalar, TSpace]:
    return field.replace(data=square_l2_norm(field).data.sqrt(), rank=Scalar)


def square_l2_norm(field: Field[Vector, TSpace]) -> Field[Scalar, TSpace]:
    return field.replace(data=(field.data.abs() ** 2).sum(dim=field.component_axes), rank=Scalar)


def square_frobenius_norm(field: Field[Tensor, TSpace]) -> Field[Scalar, TSpace]:
    return field.replace(data=(field.data.abs() ** 2).sum(dim=field.component_axes), rank=Scalar)

from specflow.core.physics.quantity.field import Field, Scalar, Tensor, TSpace, Vector


def l2_norm(field: Field[Vector, TSpace]) -> Field[Scalar, TSpace]:
    return Field(square_l2_norm(field).data.sqrt(), Scalar, field.space, field.component)


def square_l2_norm(field: Field[Vector, TSpace]) -> Field[Scalar, TSpace]:
    return Field((field.data.abs() ** 2).sum(dim=field.component_axes), Scalar, field.space, field.component)


def square_frobenius_norm(field: Field[Tensor, TSpace]) -> Field[Scalar, TSpace]:
    return Field((field.data.abs() ** 2).sum(dim=field.component_axes), Scalar, field.space, field.component)

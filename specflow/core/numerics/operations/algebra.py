from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Scalar, TRank, Tensor, TSpace, Vector


def cross_vector_vector(a: Field[Vector, TSpace], b: Field[Vector, TSpace]) -> Field[Vector, TSpace]:
    return a.replace(data=a.data.cross(b.data, dim=a.component_axes[0]))


def inner_vector_vector(a: Field[Vector, TSpace], b: Field[Vector, TSpace]) -> Field[Scalar, TSpace]:
    return a.replace(data=(a.data * b.data).sum(dim=a.component_axes), rank=Scalar)


def inner_vector_tensor(a: Field[Vector, TSpace], b: Field[Tensor, TSpace]) -> Field[Vector, TSpace]:
    return a.replace(data=(a.data.unsqueeze(b.component_axes[1]) * b.data).sum(dim=b.component_axes[0]))


def inner_tensor_tensor(a: Field[Tensor, TSpace], b: Field[Tensor, TSpace]) -> Field[Scalar, TSpace]:
    return a.replace(data=(a.data * b.data).sum(dim=a.component_axes), rank=Scalar)


def outer_vector_vector(a: Field[Vector, TSpace], b: Field[Vector, TSpace]) -> Field[Tensor, TSpace]:
    return a.replace(
        data=a.data.unsqueeze(a.component_axes[0] + 1) * b.data.unsqueeze(a.component_axes[0]), rank=Tensor
    )


def scale(field: Field[TRank, TSpace], scalar: Field[Scalar, TSpace]) -> Field[TRank, TSpace]:
    return field.replace(data=field.data * scalar.data)


def mul_scalar_scalar(a: Field[Scalar, TSpace], b: Field[Scalar, TSpace]) -> Field[Scalar, TSpace]:
    return b.replace(data=a.data * b.data)


def mul_scalar_vector(a: Field[Scalar, TSpace], b: Field[Vector, TSpace]) -> Field[Vector, TSpace]:
    return b.replace(data=a.data.unsqueeze(b.component_axes[0]) * b.data)


def mul_scalar_tensor(a: Field[Scalar, TSpace], b: Field[Tensor, TSpace]) -> Field[Tensor, TSpace]:
    return b.replace(data=a.data.unsqueeze(b.component_axes[0]).unsqueeze(b.component_axes[1]) * b.data)


def matmul_tensor_vector(a: Field[Tensor, TSpace], b: Field[Vector, TSpace]) -> Field[Vector, TSpace]:
    return b.replace(data=(a.data * b.data.unsqueeze(a.component_axes[0])).sum(dim=a.component_axes[1]))

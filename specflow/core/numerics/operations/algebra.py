from specflow.core.physics.quantity.field import Field, Scalar, Tensor, TSpace, Vector


def cross(a: Field[Vector, TSpace], b: Field[Vector, TSpace]) -> Field[Vector, TSpace]:
    return a.with_data(a.data.cross(b.data, dim=a.component_axes[0]))


def inner_vector_vector(a: Field[Vector, TSpace], b: Field[Vector, TSpace]) -> Field[Scalar, TSpace]:
    return Field((a.data * b.data).sum(dim=a.component_axes), Scalar, a.space, a.dimension)


def inner_vector_tensor(a: Field[Vector, TSpace], b: Field[Tensor, TSpace]) -> Field[Vector, TSpace]:
    return a.with_data((a.data.unsqueeze(b.component_axes[1]) * b.data).sum(dim=b.component_axes[0]))


def inner_tensor_tensor(a: Field[Tensor, TSpace], b: Field[Tensor, TSpace]) -> Field[Scalar, TSpace]:
    return Field((a.data * b.data).sum(dim=a.component_axes), Scalar, a.space, a.dimension)


def outer_vector_vector(a: Field[Vector, TSpace], b: Field[Vector, TSpace]) -> Field[Tensor, TSpace]:
    return Field(
        a.data.unsqueeze(a.component_axes[0] + 1) * b.data.unsqueeze(a.component_axes[0]), Tensor, a.space, a.dimension
    )


def mul_scalar_vector(a: Field[Scalar, TSpace], b: Field[Vector, TSpace]) -> Field[Vector, TSpace]:
    return b.with_data(a.data.unsqueeze(b.component_axes[0]) * b.data)


def mul_scalar_tensor(a: Field[Scalar, TSpace], b: Field[Tensor, TSpace]) -> Field[Tensor, TSpace]:
    return b.with_data(a.data.unsqueeze(b.component_axes[0]).unsqueeze(b.component_axes[1]) * b.data)


def matmul_tensor_vector(a: Field[Tensor, TSpace], b: Field[Vector, TSpace]) -> Field[Vector, TSpace]:
    return b.with_data(a.data * b.data.unsqueeze(a.component_axes[0]).sum(a.component_axes[1]))

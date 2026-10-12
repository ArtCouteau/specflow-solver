from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Tensor, TSpace


def transpose(field: Field[Tensor, TSpace]) -> Field[Tensor, TSpace]:
    return field.replace(data=field.data.transpose(field.component_axes[0], field.component_axes[1]))

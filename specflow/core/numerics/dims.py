

from specflow.core.numerics.context import get_context

VECTOR = (0,)
TENSOR = (0, 1)


def domain() -> tuple[int, ...]:
    return tuple(range(-get_context().config.get('grid/dimension', int), 0))

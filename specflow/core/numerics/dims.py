

from . import context

VECTOR = 0
TENSOR = (0, 1)


def domain() -> tuple[int, ...]:
    return tuple(range(-context.get().config.get('grid/dimension', int), 0))

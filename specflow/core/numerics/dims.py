import torch

from .. import context

VECTOR = 0
TENSOR = (0, 1)


def domain() -> tuple[int, ...]:
    return tuple(range(-context.get().config.get('grid/dimension', int), 0))


def view_in_domain(data: torch.Tensor) -> torch.Tensor:
    return data.view(size=(data.size() + context.get().config.get('grid/dimension', int)*(1,)))

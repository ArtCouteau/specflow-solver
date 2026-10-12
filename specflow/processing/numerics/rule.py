from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, get_type_hints

from specflow.core.physics.wrappers.wrapper import Wrapper
from specflow.processing.numerics.kind import is_concrete
from specflow.processing.numerics.node import Node

T = TypeVar("T", bound=Wrapper)


@dataclass(frozen=True)
class Rule(Generic[T]):
    output: Node[T]
    function: Callable[..., T]
    inputs: tuple[Node[Any], ...] = ()


def derive(name: str, function: Callable[..., T], *inputs: str) -> Rule[T]:
    # Binds names to a typed function: the kinds of the nodes are read from its annotations
    hints = get_type_hints(function)
    parameters = list(inspect.signature(function).parameters)
    if len(parameters) != len(inputs):
        raise TypeError(f"{function.__qualname__} takes {len(parameters)} arguments, got {len(inputs)} inputs")

    kinds = [hints["return"], *(hints[p] for p in parameters)]
    if not all(is_concrete(kind) for kind in kinds):
        raise TypeError(f"{function.__qualname__} has generic annotations, build its Rule with explicit nodes")

    return Rule(Node(name, kinds[0]), function, tuple(Node(i, k) for i, k in zip(inputs, kinds[1:])))


RULES: list[Rule[Any]] = []


def register(*rules: Rule[Any]) -> None:
    RULES.extend(rules)

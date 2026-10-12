from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from specflow.core.physics.wrappers.wrapper import Wrapper
from specflow.processing.numerics.kind import kind_tag
from specflow.processing.numerics.node import Node

T = TypeVar("T", bound=Wrapper)


class Frame:
    _values: dict[Node[Any], Wrapper]

    def __init__(self, values: Mapping[Node[Any], Wrapper] | None = None) -> None:
        self._values = {}
        for node, value in (values or {}).items():
            self[node] = value

    def get(self, node: Node[T]) -> T:
        if node not in self._values:
            raise KeyError(f"{node} not in frame, has: {[str(n) for n in self._values]}")
        return cast(T, self._values[node])

    def __setitem__(self, node: Node[T], value: T) -> None:
        if value.kind != node.kind:
            raise TypeError(f"{node} cannot hold a value of kind {kind_tag(value.kind)}")
        self._values[node] = value

    def __contains__(self, node: object) -> bool:
        return node in self._values

    def discard(self, node: Node[Any]) -> None:
        self._values.pop(node, None)

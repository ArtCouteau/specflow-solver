from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from specflow.core.physics.wrappers.wrapper import Wrapper
from specflow.processing.numerics.frame import Frame
from specflow.processing.numerics.node import Node
from specflow.processing.numerics.operators import Operator
from specflow.processing.numerics.rule import Rule


@dataclass(frozen=True)
class Step:
    output: Node[Any]
    function: Callable[..., Wrapper]
    inputs: tuple[Node[Any], ...]

    def __str__(self) -> str:
        return f"{self.output} <- {', '.join(str(n) for n in self.inputs) or '()'}"


class PostProcessor:
    _rules: tuple[Rule[Any], ...]
    _operators: tuple[Operator[Any, Any], ...]
    _inputs: frozenset[Node[Any]]
    _targets: tuple[Node[Any], ...]
    _steps: tuple[Step, ...]
    _consumers: Counter[Node[Any]]

    def __init__(
        self,
        rules: Sequence[Rule[Any]],
        operators: Sequence[Operator[Any, Any]],
        inputs: Iterable[Node[Any]],
        targets: Sequence[Node[Any]],
    ) -> None:
        self._rules = tuple(rules)
        self._operators = tuple(operators)
        self._inputs = frozenset(inputs)
        self._targets = tuple(dict.fromkeys(targets))
        self._steps = self._schedule(self._resolve())
        self._consumers = Counter(node for step in self._steps for node in step.inputs)

    @property
    def steps(self) -> tuple[Step, ...]:
        return self._steps

    def _producers(self, node: Node[Any]) -> Iterator[Step]:
        # Rules first, so that they win ties against operators
        for rule in self._rules:
            if rule.output == node:
                yield Step(node, rule.function, rule.inputs)
        for operator in self._operators:
            source = operator.source_kind(node.kind)
            if source is not None:
                yield Step(node, operator.apply, (Node(node.name, source),))

    def _resolve(self) -> dict[Node[Any], Step]:
        # Cheapest producer of each node, the cost being the number of steps to compute it from the inputs
        best: dict[Node[Any], tuple[float, Step | None]] = {node: (0.0, None) for node in self._inputs}
        failed: list[Node[Any]] = []

        def cost(node: Node[Any], visiting: frozenset[Node[Any]]) -> float:
            if node in best:
                return best[node][0]
            if node in visiting:
                return math.inf

            cheapest: tuple[float, Step | None] = (math.inf, None)
            for step in self._producers(node):
                total = 1.0
                for requirement in step.inputs:
                    total += cost(requirement, visiting | {node})
                    if math.isinf(total):
                        break
                if total < cheapest[0]:
                    cheapest = (total, step)

            # Only finite costs are memoized: an infinite one may come from the nodes being visited
            if math.isinf(cheapest[0]):
                failed.append(node)
            else:
                best[node] = cheapest
            return cheapest[0]

        for target in self._targets:
            failed.clear()
            if math.isinf(cost(target, frozenset())):
                raise ValueError(
                    f"Cannot compute {target}: no rule, operator or input produces "
                    + ", ".join(str(n) for n in dict.fromkeys(failed))
                )

        return {node: step for node, (_, step) in best.items() if step is not None}

    def _schedule(self, producers: dict[Node[Any], Step]) -> tuple[Step, ...]:
        order: list[Step] = []
        seen: set[Node[Any]] = set()

        def visit(node: Node[Any]) -> None:
            if node in seen or node in self._inputs:
                return
            seen.add(node)
            for requirement in producers[node].inputs:
                visit(requirement)
            order.append(producers[node])

        for target in self._targets:
            visit(target)

        return tuple(order)

    def run(self, inputs: Mapping[Node[Any], Wrapper]) -> dict[Node[Any], Wrapper]:
        if set(inputs) != self._inputs:
            raise KeyError(
                f"Expected inputs {sorted(str(n) for n in self._inputs)}, got {sorted(str(n) for n in inputs)}"
            )

        frame = Frame(inputs)
        pending = Counter(self._consumers)
        keep = self._inputs | set(self._targets)

        for step in self._steps:
            frame[step.output] = step.function(*(frame.get(node) for node in step.inputs))
            # Free each intermediate after its last consumer
            for node in step.inputs:
                pending[node] -= 1
                if pending[node] == 0 and node not in keep:
                    frame.discard(node)

        return {target: frame.get(target) for target in self._targets}

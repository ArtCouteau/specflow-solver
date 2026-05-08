from __future__ import annotations

from . import equation
from ..core.numerics import get_context


class Solver:
    equations: dict[str, equation.Equation]

    def __init__(self):
        get_context().generate_simulation_context()
        self.equations = {tag: equation.Equation(tag) for tag in get_context().config.get('equations', dict).keys()}

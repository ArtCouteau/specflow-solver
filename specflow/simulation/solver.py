from __future__ import annotations

from . import equation


class Solver:
    equation: equation.Equation

    def __init__(self):
        self.equation = equation.Equation()

from __future__ import annotations

import torch
from typing import Type

from .. import equation, solution
from ...core.physics import variable


class TermRegistry:
    _registry: dict[str, dict[str, Type[Term]]] = {}

    @classmethod
    def register(cls, kind: str, model: str):
        def decorator(term_class: Type[Term]) -> Type[Term]:
            if kind not in cls._registry:
                cls._registry[kind] = {}
            cls._registry[kind][model] = term_class
            return term_class
        return decorator

    @classmethod
    def get(cls, kind: str, model: str) -> Type[Term]:
        if kind not in cls._registry:
            raise KeyError(
                f'Term kind {kind} not found, available types:\n'
                + '\n'.join(f"- {k}" for k in cls._registry.keys())
            )
        if model not in cls._registry[kind]:
            raise KeyError(
                f'Term model {model} for {kind} kind not found, available models:\n'
                + '\n'.join(f"- {k}" for k in cls._registry[kind].keys())
            )
        return cls._registry[kind][model]


class Term:
    equation: equation.Equation
    tag: str
    rank: variable.Rank

    def __init__(self, equation: equation.Equation, tag: str, rank: variable.Rank):
        self.equation = equation
        self.tag = tag
        self.rank = rank

    def compute(self, variable: solution.SolutionVariable, time: float) -> torch.Tensor: ...

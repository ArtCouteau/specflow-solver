from __future__ import annotations

import torch
from typing import Type

from specflow.core.physics import field
from specflow.simulation import solver


class TermRegistry:
    _registry: dict[str, dict[str, Type[Term]]] = {}

    @classmethod
    def register(cls, kind: str, model: str):
        def decorator(term_class:  Type[Term]) -> Type[Term]:
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
    kind: str
    model: str
    rank: field.Rank

    def __init__(self):
        pass

    def setup(self, solver: solver.Solver) -> None: ...

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor: ...

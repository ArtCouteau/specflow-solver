from __future__ import annotations

import torch
from typing import Type


class TermRegistry:
    _registry: dict[str, dict[str, Type[Term]]] = {}

    @classmethod
    def register(cls, type: str, model: str):
        def decorator(term_class:  Type[Term]) -> Type[Term]:
            if type not in cls._registry:
                cls._registry[type] = {}
            cls._registry[type][model] = term_class
            return term_class
        return decorator

    @classmethod
    def get(cls, type: str, model: str) -> Type[Term]:
        if type not in cls._registry:
            raise KeyError(
                f'Term type {type} not found, available types:\n'
                + '\n'.join(f"- {k}" for k in cls._registry.keys())
            )
        if model not in cls._registry[type]:
            raise KeyError(
                f'Term model {model} for {type} type not found, available models:\n'
                + '\n'.join(f"- {k}" for k in cls._registry[type].keys())
            )
        return cls._registry[type][model]


class Term:

    def __init__(self):
        pass

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor: ...

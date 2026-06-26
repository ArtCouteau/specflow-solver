from __future__ import annotations

import torch


class SolutionVariable:
    data: dict[str, torch.Tensor]

    def __init__(self, data: dict[str, torch.Tensor]):
        self.data = data

    def get(self, key: str) -> torch.Tensor:
        return self.data[key]

    def add_(self, other: SolutionVariable, alpha: float = 1.0) -> SolutionVariable:
        for key in self.data:
            self.data[key].add_(other.data[key], alpha=alpha)
        return self

    def mul_(self, scalar: int | float) -> SolutionVariable:
        for v in self.data.values():
            v.mul_(scalar)
        return self

    def copy_(self, other: SolutionVariable) -> SolutionVariable:
        for key in self.data:
            self.data[key].copy_(other.data[key])
        return self

    def zero_(self) -> SolutionVariable:
        for key in self.data:
            self.data[key].zero_()
        return self

    def __add__(self, other: SolutionVariable) -> SolutionVariable:
        return SolutionVariable({k: v + other.data[k] for k, v in self.data.items()})

    def __mul__(self, scalar: int | float) -> SolutionVariable:
        return SolutionVariable({k: v * scalar for k, v in self.data.items()})

    def __rmul__(self, scalar: int | float) -> SolutionVariable:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: int | float) -> SolutionVariable:
        return SolutionVariable({k: v / scalar for k, v in self.data.items()})

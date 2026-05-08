from __future__ import annotations
import math

import torch
from typing import Callable, Type

from specflow.core.numerics.context import get_context
from specflow.core.physics import dims, fourier, norm


class DealiasRegistry:
    _registry: dict[str, Type[Dealias]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(dealiasing: Type[Dealias]) -> Type[Dealias]:
            cls._registry[name] = dealiasing
            return dealiasing
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[Dealias]:
        if not name in cls._registry:
            raise KeyError(
                f'Dealias \'{name}\' not found, available:\n'
                + '\n'.join(f"- {k}" for k in cls._registry.keys())
            )
        return cls._registry[name]


class Dealias:
    def _filter(self) -> torch.Tensor: ...

    def _pseudo_spectral_pass(
        self,
        function: Callable[..., torch.Tensor],
        transform_forward: Callable[[torch.Tensor], torch.Tensor],
        transform_backward: Callable[[torch.Tensor], torch.Tensor],
        *data: torch.Tensor
    ) -> torch.Tensor:
        return transform_backward(function(*(transform_forward(d) for d in data)))

    def apply(self, function: Callable[..., torch.Tensor], *data: torch.Tensor) -> torch.Tensor: ...


@DealiasRegistry.register('smoothing')
class SmoothingDealias(Dealias):
    order: int

    def __init__(self):
        get_context().config.get('schemes/dealias/order', int, 36)

    def _filter(self) -> torch.Tensor:
        return torch.exp(self.order * norm.l2_norm(get_context().mesh.grid)**self.order)

    def apply(self, function: Callable[..., torch.Tensor], *data: torch.Tensor) -> torch.Tensor:
        return (
            self._filter()
            * self._pseudo_spectral_pass(function, fourier.to_spectral, fourier.to_physical, *data)
        )


@DealiasRegistry.register('2/3')
class TwoThirdsDealias(Dealias):
    def _filter(self) -> torch.Tensor:
        return (norm.square_l2_norm(get_context().mesh.grid) < (get_context().config.get('grid/N', int)/3.0)**2)

    def apply(self, function: Callable[..., torch.Tensor], *data: torch.Tensor) -> torch.Tensor:
        return (
            self._filter()
            * self._pseudo_spectral_pass(function, fourier.to_spectral, fourier.to_physical, *data)
        )


@DealiasRegistry.register('phaseshift')
class PhaseshiftDealias(Dealias):
    def _shift(self, data: torch.Tensor) -> torch.Tensor:
        return (
            data
            * torch.exp(
                1.0j
                * get_context().mesh.grid.sum(dim=dims.VECTOR)
                * math.pi/get_context().config.get('grid/N', int)
            )
        )

    def _inverse_shift(self, data: torch.Tensor) -> torch.Tensor:
        return (
            data
            * torch.exp(
                - 1.0j
                * get_context().mesh.grid.sum(dim=dims.VECTOR)
                * math.pi/get_context().config.get('grid/N', int)
            )
        )

    def _filter(self) -> torch.Tensor:
        return (norm.square_l2_norm(get_context().mesh.grid) < (get_context().config.get('grid/N', int)/3.0)**2)

    def _transform_backward(self, data: torch.Tensor) -> torch.Tensor:
        return self._inverse_shift(fourier.to_spectral(data))

    def _transform_forward(self, data: torch.Tensor) -> torch.Tensor:
        return fourier.to_physical(self._shift(data))

    def apply(self, function: Callable[..., torch.Tensor], *data: torch.Tensor) -> torch.Tensor:
        return (
            self._filter()
            * 0.5*(
                self._pseudo_spectral_pass(function, fourier.to_spectral, fourier.to_physical, *data)
                + self._pseudo_spectral_pass(function, self._transform_forward, self._transform_backward, *data)
            )
        )

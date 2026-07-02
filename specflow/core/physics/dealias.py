from __future__ import annotations

import math
import torch
from typing import Callable

from .. import context
from ..numerics.operations import fourier, mask, norm
from ..numerics import dims
from ..registry import Registry


class Dealias:
    filter: torch.Tensor

    def _pseudo_spectral_pass(
        self,
        function: Callable[..., torch.Tensor],
        transform_forward: Callable[[torch.Tensor], torch.Tensor],
        transform_backward: Callable[[torch.Tensor], torch.Tensor],
        *data: torch.Tensor
    ) -> torch.Tensor:
        return transform_forward(function(*(transform_backward(d) for d in data)))

    def apply(self, function: Callable[..., torch.Tensor], *data: torch.Tensor) -> torch.Tensor: ...


class DealiasRegistry(Registry[Dealias]):
    pass


@DealiasRegistry.register('smoothing')
class SmoothingDealias(Dealias):
    order: int

    def __init__(self):
        self.order = context.get().config.get('schemes/dealias/order', int, 36)
        kMax = context.get().config.get('grid/N', int)//2
        self.filter = torch.exp(- self.order * (norm.l2_norm(context.get().mesh.grid)/kMax)**self.order)

    def apply(self, function: Callable[..., torch.Tensor], *data: torch.Tensor) -> torch.Tensor:
        return (
            self.filter
            * self._pseudo_spectral_pass(function, fourier.to_spectral, fourier.to_physical, *data)
        )


@DealiasRegistry.register('2/3')
class TwoThirdsDealias(Dealias):
    def __init__(self):
        self.filter = mask.spherical(context.get().config.get('grid/N', int)/3.0)

    def apply(self, function: Callable[..., torch.Tensor], *data: torch.Tensor) -> torch.Tensor:
        return (
            self.filter
            * self._pseudo_spectral_pass(function, fourier.to_spectral, fourier.to_physical, *data)
        )


@DealiasRegistry.register('phaseshift')
class PhaseshiftDealias(Dealias):
    _shiftFactor: torch.Tensor
    _inverseShiftFactor: torch.Tensor

    def __init__(self):
        phase = (
            context.get().mesh.grid.sum(dim=dims.VECTOR)
            * math.pi/context.get().config.get('grid/N', int)
        )
        self._shiftFactor = torch.exp(1.0j*phase)
        self._inverseShiftFactor = torch.exp(-1.0j*phase)
        self.filter = mask.spherical(context.get().config.get('grid/N', int)/2.0)

    def _shift(self, data: torch.Tensor) -> torch.Tensor:
        return data * self._shiftFactor

    def _inverse_shift(self, data: torch.Tensor) -> torch.Tensor:
        return data * self._inverseShiftFactor

    def _transform_backward(self, data: torch.Tensor) -> torch.Tensor:
        return fourier.to_physical(self._shift(data))

    def _transform_forward(self, data: torch.Tensor) -> torch.Tensor:
        return self._inverse_shift(fourier.to_spectral(data))

    def apply(self, function: Callable[..., torch.Tensor], *data: torch.Tensor) -> torch.Tensor:
        return (
            self.filter
            * 0.5*(
                self._pseudo_spectral_pass(function, fourier.to_spectral, fourier.to_physical, *data)
                + self._pseudo_spectral_pass(function, self._transform_forward, self._transform_backward, *data)
            )
        )

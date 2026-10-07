from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar, override

import torch

from specflow.core.numerics.mesh import Mesh
from specflow.core.numerics.operations import algebra, filters, fourier
from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Physical, Rank, Scalar, Spectral

RankA = TypeVar("RankA", bound=Rank)
RankB = TypeVar("RankB", bound=Rank)
RankOut = TypeVar("RankOut", bound=Rank)
TDealias = TypeVar("TDealias", bound=Rank)


class QuadraticDealias(ABC):
    _filter: Field[Scalar, Spectral]

    def _pseudo_spectral_pass(
        self,
        operation: Callable[[Field[RankA, Physical], Field[RankB, Physical]], Field[RankOut, Physical]],
        a: Field[RankA, Spectral],
        b: Field[RankB, Spectral],
    ) -> Field[RankOut, Spectral]:
        return fourier.to_spectral(operation(fourier.to_physical(a), fourier.to_physical(b)))

    @abstractmethod
    def apply(
        self,
        operation: Callable[[Field[RankA, Physical], Field[RankB, Physical]], Field[RankOut, Physical]],
        a: Field[RankA, Spectral],
        b: Field[RankB, Spectral],
    ) -> Field[RankOut, Spectral]: ...


class TwoThirdsDealias(QuadraticDealias):
    def __init__(self, mesh: Mesh):
        self._filter = filters.spherical_truncation(mesh, mesh.layout.N / 3.0)

    @override
    def apply(
        self,
        operation: Callable[[Field[RankA, Physical], Field[RankB, Physical]], Field[RankOut, Physical]],
        a: Field[RankA, Spectral],
        b: Field[RankB, Spectral],
    ) -> Field[RankOut, Spectral]:
        return self._pseudo_spectral_pass(operation, a, b)


class PhaseshiftDealias(QuadraticDealias):
    _shift: Field[Scalar, Spectral]
    _unshift: Field[Scalar, Spectral]

    def __init__(self, mesh: Mesh):
        k = mesh.spectral.wavenumbers
        phase = k.data.sum(dim=k.component_axes) * (0.5 * mesh.physical.spacing)
        self._shift = k.replace(data=torch.exp(1.0j * phase), rank=Scalar)
        self._unshift = self._shift.replace(data=torch.exp(-1.0j * phase))
        self._filter = filters.spherical_truncation(mesh, mesh.layout.N * math.sqrt(2.0) / 3.0)

    @override
    def apply(
        self,
        operation: Callable[[Field[RankA, Physical], Field[RankB, Physical]], Field[RankOut, Physical]],
        a: Field[RankA, Spectral],
        b: Field[RankB, Spectral],
    ) -> Field[RankOut, Spectral]:
        return self._pseudo_spectral_pass(operation, a, b).add_(
            algebra.scale(
                self._pseudo_spectral_pass(operation, algebra.scale(a, self._shift), algebra.scale(b, self._shift)),
                self._unshift,
            )
        )


class FourierSmoothing:
    order: int
    _filter: Field[Scalar, Spectral]

    def __init__(self, mesh: Mesh, order: int = 32):
        self.order = order
        kNorm = mesh.spectral.wavenumbersSquared.data.sqrt() / mesh.spectral.maxWavenumber
        self._filter = mesh.spectral.wavenumbersSquared.replace(data=torch.exp(-order * kNorm**order))

    def _pseudo_spectral_pass(
        self, operation: Callable[[Field[RankA, Physical]], Field[RankOut, Physical]], a: Field[RankA, Spectral]
    ) -> Field[RankOut, Spectral]:
        return fourier.to_spectral(operation(fourier.to_physical(a)))

    def apply(
        self, operation: Callable[[Field[RankA, Physical]], Field[RankOut, Physical]], a: Field[RankA, Spectral]
    ) -> Field[RankOut, Spectral]:
        return algebra.scale(self._pseudo_spectral_pass(operation, a), self._filter)

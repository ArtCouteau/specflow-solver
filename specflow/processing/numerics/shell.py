from __future__ import annotations

import functools
import math

import torch

from specflow.core.numerics.mesh import Mesh


class SpectralShells:
    bounds: torch.Tensor
    wavenumbers: torch.Tensor
    index: torch.Tensor
    smoothing: torch.Tensor

    def __init__(self, mesh: Mesh) -> None:
        # Shell i holds the modes with (i - 1/2) k0 < |k| <= (i + 1/2) k0, clipped to (0, kMax]
        k0 = 2.0 * math.pi / mesh.layout.LENGTH
        kMax = mesh.layout.N // 2
        kSquared = mesh.spectral.wavenumbersSquared.data

        edges = torch.arange(-0.5, kMax + 1.5, dtype=kSquared.dtype, device=kSquared.device)
        edges[0] = 0.0
        edges[-1] = kMax
        self.bounds = k0 * edges
        self.wavenumbers = k0 * torch.arange(kMax + 1, dtype=kSquared.dtype, device=kSquared.device)

        # Modes outside every shell (k = 0, |k| > kMax) go to the extra bin nShells, dropped when summing
        index = torch.bucketize(kSquared, self.bounds**2) - 1
        self.index = torch.where((index >= 0) & (index < self.nShells), index, self.nShells).flatten()

        self.smoothing = self._compute_smoothing_factor(mesh, edges)

    @property
    def nShells(self) -> int:
        return self.bounds.size(0) - 1

    def sum(self, data: torch.Tensor, dimension: int) -> torch.Tensor:
        # Sums each shell over the trailing `dimension` domain axes of data
        flat = data.flatten(start_dim=data.dim() - dimension)
        shells = flat.new_zeros((*flat.shape[:-1], self.nShells + 1))
        shells.index_add_(flat.dim() - 1, self.index, flat)
        return shells[..., :-1]

    def _compute_smoothing_factor(self, mesh: Mesh, edges: torch.Tensor) -> torch.Tensor:
        counts = self.sum(mesh.spectral.parsevalWeights.data, mesh.layout.DIMENSION)
        density = counts.sum() / ((4.0 / 3.0) * math.pi * edges[-1] ** 3)
        volumes = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

        smoothing = torch.zeros_like(volumes)
        smoothing[1:] = volumes[1:] / counts[1:] * density
        return smoothing


@functools.cache
def spectral_shells(mesh: Mesh) -> SpectralShells:
    return SpectralShells(mesh)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from specflow.core.numerics.backend.backend import NumericsBackend
from specflow.core.numerics.operations import norm
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, Scalar, Spectral, Vector


@dataclass(frozen=True)
class MeshLayout:
    N: int
    LENGTH: float
    DIMENSION: Literal[1, 2, 3]


@dataclass(frozen=True)
class Mesh:
    layout: MeshLayout
    physical: PhysicalGrid
    spectral: SpectralGrid


class SpectralGrid:
    wavenumbers: Field[Vector, Spectral]
    maxWavenumber: float
    wavenumbersSquared: Field[Scalar, Spectral]
    inverseWavenumbersSquared: Field[Scalar, Spectral]
    parsevalWeights: Field[Scalar, Spectral]
    domainShape: tuple[int, ...]

    def __init__(self, mesh: Mesh, numerics: NumericsBackend) -> None:
        self.wavenumbers = Field(
            2.0 * math.pi / mesh.layout.LENGTH * self._construct_grid(mesh.layout, numerics),
            Vector,
            Spectral,
            mesh.layout.DIMENSION,
            mesh.layout.DIMENSION,
            numerics,
            mesh,
        )

        self.wavenumbersSquared = Field(
            norm.square_l2_norm(self.wavenumbers).data,
            Scalar,
            Spectral,
            1,
            mesh.layout.DIMENSION,
            numerics,
            mesh,
        )

        self.inverseWavenumbersSquared = Field(
            torch.where(self.wavenumbersSquared.data > 0, 1 / self.wavenumbersSquared.data, 0.0),
            Scalar,
            Spectral,
            1,
            mesh.layout.DIMENSION,
            numerics,
            mesh,
        )

        self.parsevalWeights = Field(
            torch.empty_like(self.wavenumbersSquared.data),
            Scalar,
            Spectral,
            1,
            mesh.layout.DIMENSION,
            numerics,
            mesh,
        )
        self.parsevalWeights.data.fill_(2.0)
        self.parsevalWeights.data[..., 0] = 1.0
        self.parsevalWeights.data[..., -1] = 1.0

        self.domainShape = (mesh.layout.DIMENSION - 1) * (mesh.layout.N,) + (mesh.layout.N // 2 + 1,)
        self.maxWavenumber = 2.0 * math.pi / mesh.layout.LENGTH * (mesh.layout.N // 2)

    def _construct_grid(self, layout: MeshLayout, numerics: NumericsBackend) -> torch.Tensor:
        return _construct_grid(
            layout,
            torch.fft.fftfreq(layout.N, 1.0 / layout.N),
            torch.fft.rfftfreq(layout.N, 1.0 / layout.N),
        ).to(dtype=numerics.parameters.precision.INT, device=numerics.parameters.device)


class PhysicalGrid:
    grid: Field[Vector, Physical]
    spacing: float
    domainShape: tuple[int, ...]

    def __init__(self, mesh: Mesh, numerics: NumericsBackend) -> None:
        self.grid = Field(
            self._construct_grid(mesh.layout, numerics),
            Vector,
            Physical,
            mesh.layout.DIMENSION,
            mesh.layout.DIMENSION,
            numerics,
            mesh,
        )

        self.spacing = mesh.layout.LENGTH / mesh.layout.N
        self.domainShape = (mesh.layout.DIMENSION) * (mesh.layout.N,)

    def _construct_grid(self, layout: MeshLayout, numerics: NumericsBackend) -> torch.Tensor:
        tmp = layout.LENGTH / layout.N * torch.arange(0, layout.N)

        return _construct_grid(layout, tmp, tmp).to(
            dtype=numerics.parameters.precision.FLOAT,
            device=numerics.parameters.device,
        )


def _construct_grid(layout: MeshLayout, dim: torch.Tensor, dimLast: torch.Tensor) -> torch.Tensor:
    if layout.DIMENSION == 1:
        _grid = dimLast.unsqueeze(0)
    elif layout.DIMENSION == 2:
        _grid = torch.stack(torch.meshgrid(dim, dimLast, indexing="ij"))
    elif layout.DIMENSION == 3:
        _grid = torch.stack(torch.meshgrid(dim, dim, dimLast, indexing="ij"))
    else:
        raise ValueError(f"Invalid grid dimension: {layout.DIMENSION} (expected 1, 2 or 3)")

    return _grid

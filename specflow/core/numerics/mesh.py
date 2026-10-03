from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from specflow.core.numerics.backend.backend import NumericsBackend
from specflow.core.numerics.operations import norm
from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Physical, Scalar, Spectral, Vector


@dataclass(frozen=True)
class MeshLayout:
    N: int
    LENGTH: float
    DIMENSION: Literal[1, 2, 3]


@dataclass(frozen=True)
class Mesh:
    spectral: SpectralGrid
    physical: PhysicalGrid


class SpectralGrid:
    wavenumbers: Field[Vector, Spectral]
    wavenumbersSquared: Field[Scalar, Spectral]
    inverseWavenumbersSquared: Field[Scalar, Spectral]
    rfftWeights: Field[Scalar, Spectral]

    def __init__(self, layout: MeshLayout, numerics: NumericsBackend) -> None:
        self.wavenumbers = Field(
            numerics.memory.alloc(
                "mesh/spectral/wavenumbers",
                (layout.DIMENSION,) + _domain_shape(layout),
                dtype=numerics.parameters.precision.FLOAT,
                device=numerics.parameters.device,
            ),
            Vector,
            Spectral,
            layout.DIMENSION,
            layout.DIMENSION,
        )
        self.wavenumbers.data.copy_(2.0 * torch.pi / layout.LENGTH * self._construct_grid(layout, numerics))

        self.wavenumbersSquared = Field(
            numerics.memory.alloc(
                "mesh/spectral/wavenumbersSquared",
                _domain_shape(layout),
                dtype=numerics.parameters.precision.FLOAT,
                device=numerics.parameters.device,
            ),
            Scalar,
            Spectral,
            1,
            layout.DIMENSION,
        )
        self.wavenumbersSquared.copy_(norm.square_l2_norm(self.wavenumbers))

        self.inverseWavenumbersSquared = Field(
            numerics.memory.alloc(
                "mesh/spectral/inverseWavenumbersSquared",
                _domain_shape(layout),
                dtype=numerics.parameters.precision.FLOAT,
                device=numerics.parameters.device,
            ),
            Scalar,
            Spectral,
            1,
            layout.DIMENSION,
        )
        self.wavenumbersSquared.data.copy_(
            torch.where(self.wavenumbersSquared.data > 0, 1 / self.wavenumbersSquared.data, 0.0)
        )

    def _construct_grid(self, layout: MeshLayout, numerics: NumericsBackend) -> torch.Tensor:
        return _construct_grid(
            layout,
            torch.fft.fftfreq(layout.N, 1.0 / layout.N),
            torch.fft.rfftfreq(layout.N, 1.0 / layout.N),
        ).to(dtype=numerics.parameters.precision.INT, device=numerics.parameters.device)


class PhysicalGrid:
    grid: Field[Vector, Physical]

    def __init__(self, layout: MeshLayout, numerics: NumericsBackend) -> None:
        self.grid = Field(
            numerics.memory.alloc(
                "mesh/physical/grid",
                (layout.DIMENSION,) + _domain_shape(layout),
                dtype=numerics.parameters.precision.FLOAT,
                device=numerics.parameters.device,
            ),
            Vector,
            Physical,
            layout.DIMENSION,
            layout.DIMENSION,
        )
        self.grid.data.copy_(2.0 * torch.pi / layout.LENGTH * self._construct_grid(layout, numerics))

    def _construct_grid(self, layout: MeshLayout, numerics: NumericsBackend) -> torch.Tensor:
        tmp = layout.LENGTH / layout.N * torch.arange(0, layout.N)

        return _construct_grid(layout, tmp, tmp).to(
            dtype=numerics.parameters.precision.FLOAT,
            device=numerics.parameters.device,
        )


def _domain_shape(layout: MeshLayout) -> tuple[int, ...]:
    return (layout.DIMENSION - 1) * (layout.N,) + (layout.N // 2 + 1,)


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

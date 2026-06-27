from __future__ import annotations

import torch

from . import context
from .decomposition import communication
from ..physics import variable


class Mesh:
    grid: torch.Tensor
    k: torch.Tensor
    k2: torch.Tensor

    def __init__(self):
        self.grid = context.get().memory.alloc(
            'mesh',
            variable.field_shape(variable.Rank(1)),
            dtype=torch.int64,
            device=context.get().config.get('numerics/device', str)
        )
        self.grid.copy_(construct_spectral_grid())

        self.k = context.get().memory.alloc(
            'mesh/k',
            variable.field_shape(variable.Rank(1)),
            dtype=context.get().config.get('numerics/precision/float', torch.dtype),
            device=context.get().config.get('numerics/device', str)
        )
        self.k.copy_(2.0*torch.pi/context.get().config.get('grid/length', float)*self.grid)

        self.k2 = context.get().memory.alloc(
            'mesh/k2',
            variable.field_shape(variable.Rank(0)),
            dtype=context.get().config.get('numerics/precision/float', torch.dtype),
            device=context.get().config.get('numerics/device', str)
        )
        self.k2.copy_(self.k.pow(2).sum(dim=0))

    def physical_grid_spacing(self) -> float:
        return context.get().config.get('grid/length', float)/context.get().config.get('grid/N', int)


def _construct_grid(dimFirst: torch.Tensor, dimMiddle: torch.Tensor, dimLast: torch.Tensor) -> torch.Tensor:
    DIM: int = context.get().config.get('grid/dimension', int)

    if DIM == 1:
        _grid = dimLast.unsqueeze(0)
    elif DIM == 2:
        _grid = torch.stack(torch.meshgrid(dimFirst, dimLast, indexing="ij"))
    elif DIM == 3:
        _grid = torch.stack(torch.meshgrid(dimFirst, dimMiddle, dimLast, indexing="ij"))
    else:
        raise ValueError(f'Invalid grid dimension: {DIM} (expected 1, 2 or 3)')

    return _grid


def _local_split_axis(freq: torch.Tensor) -> torch.Tensor:
    if not context.get().distribution.enabled:
        return freq

    start, end = communication.local_slice(freq.numel())
    return freq[start:end]


def construct_spectral_grid() -> torch.Tensor:
    N: int = context.get().config.get('grid/N', int)

    dimFull = torch.fft.fftfreq(N, 1.0/N)    # type: ignore

    return _construct_grid(
        _local_split_axis(dimFull),
        dimFull,
        torch.fft.rfftfreq(N, 1.0/N)    # type: ignore
    ).to(
        dtype=torch.int64,
        device=context.get().config.get('numerics/device', str)
    )


def construct_full_spectral_grid() -> torch.Tensor:
    N: int = context.get().config.get('grid/N', int)

    dimFull = torch.fft.fftfreq(N, 1.0/N)       # type: ignore
    return _construct_grid(
        _local_split_axis(dimFull),
        dimFull,
        dimFull
    ).to(    # type: ignore
        dtype=torch.int64,
        device=context.get().config.get('numerics/device', str)
    )


def construct_physical_grid() -> torch.Tensor:
    L: float = context.get().config.get('grid/length', float)
    N: int = context.get().config.get('grid/N', int)

    tmp = L/N*torch.arange(0, N)

    return _construct_grid(tmp, tmp, tmp).to(
        dtype=context.get().config.get('numerics/precision/float', torch.dtype),
        device=context.get().config.get('numerics/device', str)
    )


def get_physical_grid() -> torch.Tensor:
    return construct_physical_grid()


def get_spectral_grid() -> torch.Tensor:
    return construct_spectral_grid()

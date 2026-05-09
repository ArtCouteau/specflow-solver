from __future__ import annotations

import torch

from . import get_context
from ..physics import variable


class Mesh:
    grid: torch.Tensor

    def __init__(self):
        self.grid = get_context().memory.alloc('mesh', variable.field_shape(variable.Rank(1)), dtype=torch.int64)
        self.grid.copy_(construct_spectral_grid())

    def domain_shape(self) -> tuple[int, ...]:
        N: int = get_context().config.get('grid/N', int)
        DIM: int = get_context().config.get('grid/dimension', int)

        return (N,) * (DIM - 1) + (N // 2 + 1,)


def _construct_grid(dim: torch.Tensor, dimLast: torch.Tensor) -> torch.Tensor:
    DIM: int = get_context().config.get('grid/dimension', int)

    _grid: torch.Tensor = torch.empty()
    if DIM == 1:
        _grid = dimLast.unsqueeze(0)
    elif DIM == 2:
        _grid = torch.stack(torch.meshgrid(dim, dimLast, indexing="ij"))
    elif DIM == 3:
        _grid = torch.stack(torch.meshgrid(dim, dim, dimLast, indexing="ij"))
    else:
        raise ValueError(f'Invalid grid dimension: {DIM} (expected 1, 2 or 3)')

    return _grid


def construct_spectral_grid() -> torch.Tensor:
    N: int = get_context().config.get('grid/N', int)

    return _construct_grid(
        torch.fft.fftfreq(N, 1.0/N),    # type: ignore
        torch.fft.rfftfreq(N, 1.0/N)    # type: ignore
    ).to(
        dtype=torch.int64,
        device=get_context().config.get('numerics/device', str)
    )


def construct_full_spectral_grid() -> torch.Tensor:
    N: int = get_context().config.get('grid/N', int)

    tmp = torch.fft.fftfreq(N, 1.0/N)       # type: ignore
    return _construct_grid(tmp, tmp).to(    # type: ignore
        dtype=torch.int64,
        device=get_context().config.get('numerics/device', str)
    )


def construct_physical_grid() -> torch.Tensor:
    L: float = get_context().config.get('grid/length', float)
    N: int = get_context().config.get('grid/N', int)

    tmp = L/N*torch.torch.arange(0, N)

    return _construct_grid(tmp, tmp).to(
        dtype=get_context().config.get('numerics/precision/float', torch.dtype),
        device=get_context().config.get('numerics/device', str)
    )

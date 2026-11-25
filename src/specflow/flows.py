"""Initial conditions and exact solutions shared by tests and cases."""

import math
from collections.abc import Mapping

import torch

from specflow.mesh import Mesh


def diffusion_1d_physical(
    mesh: Mesh, amplitudes: Mapping[int, float], diffusivity: float, time: float
) -> torch.Tensor:
    """Return the exact solution Σₙ aₙ sin(kₙx) exp(−κkₙ²t) of 1D diffusion, (N,).

    Args:
        mesh: A 1D mesh.
        amplitudes: The amplitude aₙ of each integer wavenumber n, with kₙ = 2πn/L.
        diffusivity: κ.
        time: t.
    """
    if mesh.dimension != 1:
        raise ValueError(f"1D diffusion needs a 1D mesh, got {mesh.dimension}D")
    position = mesh.coordinates[0]
    solution_physical = torch.zeros_like(position)
    for integer, amplitude in amplitudes.items():
        wavenumber = 2 * math.pi * integer / mesh.length
        decay = math.exp(-diffusivity * wavenumber**2 * time)
        solution_physical.add_(
            torch.sin(wavenumber * position), alpha=amplitude * decay
        )
    return solution_physical

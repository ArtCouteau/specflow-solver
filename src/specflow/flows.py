"""Initial conditions and exact solutions shared by tests and cases."""

import math
from collections.abc import Mapping

import torch

from specflow.mesh import Mesh


def diffusion_1d_physical(
    mesh: Mesh, amplitudes: Mapping[int, float], diffusivity: float, time: float
) -> torch.Tensor:
    r"""Return the exact solution of 1D diffusion from a sum of sines.

    The solution is $\sum_n a_n \sin(\kappa_n x) e^{-\alpha \kappa_n^2 t}$, with
    $\kappa_n = 2 \pi n / L$ and ``amplitudes`` mapping each $n$ to $a_n$.
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


def taylor_green_2d_physical(mesh: Mesh, viscosity: float, time: float) -> torch.Tensor:
    r"""Return the decaying 2D Taylor-Green vortex.

    The velocity is
    $(\sin \kappa_0 x \cos \kappa_0 y, -\cos \kappa_0 x \sin \kappa_0 y)
    e^{-2 \nu \kappa_0^2 t}$.
    """
    if mesh.dimension != 2:
        raise ValueError(f"2D Taylor-Green needs a 2D mesh, got {mesh.dimension}D")
    wavenumber = 2 * math.pi / mesh.length
    decay = math.exp(-2 * viscosity * wavenumber**2 * time)
    x, y = mesh.coordinates.mul(wavenumber)
    velocity_physical = torch.stack(
        [torch.sin(x) * torch.cos(y), torch.cos(x) * torch.sin(y).neg_()]
    )
    return velocity_physical.mul_(decay)


def taylor_green_3d_physical(mesh: Mesh) -> torch.Tensor:
    r"""Return the initial 3D Taylor-Green vortex.

    The velocity is
    $(\sin \kappa_0 x \cos \kappa_0 y \cos \kappa_0 z,
    -\cos \kappa_0 x \sin \kappa_0 y \cos \kappa_0 z, 0)$.
    """
    if mesh.dimension != 3:
        raise ValueError(f"3D Taylor-Green needs a 3D mesh, got {mesh.dimension}D")
    x, y, z = mesh.coordinates.mul(2 * math.pi / mesh.length)
    cos_z = torch.cos(z)
    velocity_physical = torch.zeros_like(mesh.coordinates)
    velocity_physical[0] = torch.sin(x).mul_(torch.cos(y)).mul_(cos_z)
    velocity_physical[1] = torch.cos(x).mul_(torch.sin(y)).mul_(cos_z).neg_()
    return velocity_physical

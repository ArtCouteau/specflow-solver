"""Global diagnostics of spectral fields, as Parseval sums with no FFT.

Each diagnostic reduces the component and spatial axes through the mesh's
decomposition and keeps the leading axes, in the mesh's dtype and on its device.
⟨·⟩ is the mean over the periodic domain.
"""

import torch

from specflow.mesh import Mesh


def _mean_square(
    mesh: Mesh,
    field_spectral: torch.Tensor,
    component_axes: int = 0,
    multiplier: torch.Tensor | None = None,
) -> torch.Tensor:
    density_spectral = field_spectral.abs().square_().mul_(mesh.parseval_weights)
    if multiplier is not None:
        density_spectral.mul_(multiplier)
    if component_axes:
        component_dims = tuple(range(-mesh.dimension - component_axes, -mesh.dimension))
        density_spectral = density_spectral.sum(dim=component_dims)
    return mesh.decomposition.sum(density_spectral)


def kinetic_energy(mesh: Mesh, velocity_spectral: torch.Tensor) -> torch.Tensor:
    """Return the kinetic energy ½⟨|u|²⟩."""
    return _mean_square(mesh, velocity_spectral, 1).mul_(0.5)


def dissipation(
    mesh: Mesh, velocity_spectral: torch.Tensor, viscosity: float
) -> torch.Tensor:
    """Return the dissipation ν⟨∂ⱼuᵢ∂ⱼuᵢ⟩, as −ν⟨u·∇²u⟩.

    Built on the exact k² of the Laplacian, so that it is minus the energy
    contribution of diffusion.
    """
    return _mean_square(mesh, velocity_spectral, 1, mesh.wavenumbers_squared).mul_(
        viscosity
    )


def enstrophy(mesh: Mesh, velocity_spectral: torch.Tensor) -> torch.Tensor:
    """Return the enstrophy ½⟨|ω|²⟩ of a 2D or 3D velocity."""
    if mesh.dimension == 2:
        return _mean_square(mesh, mesh.curl_2d(velocity_spectral)).mul_(0.5)
    if mesh.dimension == 3:
        return _mean_square(mesh, mesh.curl_3d(velocity_spectral), 1).mul_(0.5)
    raise ValueError(f"enstrophy needs a 2D or 3D mesh, got {mesh.dimension}D")


def divergence_rms(mesh: Mesh, velocity_spectral: torch.Tensor) -> torch.Tensor:
    """Return the RMS divergence ⟨(∇·u)²⟩^½."""
    return _mean_square(mesh, mesh.divergence(velocity_spectral)).sqrt_()


def relative_error(
    mesh: Mesh,
    field_spectral: torch.Tensor,
    reference_spectral: torch.Tensor,
    component_axes: int = 0,
) -> torch.Tensor:
    """Return the relative L2 error ‖f − r‖/‖r‖ of a field against a reference.

    Args:
        mesh: The mesh of both fields.
        field_spectral: f.
        reference_spectral: r, with the same layout as f.
        component_axes: The number of component axes, 0 for a scalar and 1 for a
            vector; they are summed with the spatial axes.
    """
    difference = _mean_square(mesh, field_spectral - reference_spectral, component_axes)
    reference = _mean_square(mesh, reference_spectral, component_axes)
    return difference.div_(reference).sqrt_()

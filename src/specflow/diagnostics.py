"""Diagnostics of a spectral velocity, and the local fields they average.

Each global diagnostic is a Parseval sum with no FFT; it reduces the component and
spatial axes and keeps the leading axes. Each local field is built at the mesh points,
for the moments and histograms of the distributions module.
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
    r"""Return the kinetic energy $\tfrac12 \langle u_i u_i \rangle$."""
    return _mean_square(mesh, velocity_spectral, 1).mul_(0.5)


def fluctuation_kinetic_energy(
    mesh: Mesh, velocity_spectral: torch.Tensor
) -> torch.Tensor:
    r"""Return the kinetic energy of the fluctuation, without the mean mode.

    It is $\tfrac12 \langle u'_i u'_i \rangle$.
    """
    fluctuating = (mesh.wavenumbers_squared != 0).to(mesh.dtype)
    return _mean_square(mesh, velocity_spectral, 1, fluctuating).mul_(0.5)


def dissipation(
    mesh: Mesh, velocity_spectral: torch.Tensor, viscosity: float
) -> torch.Tensor:
    r"""Return the dissipation $\nu \langle \partial_j u_i \partial_j u_i \rangle$.

    It is exactly minus the energy contribution of diffusion,
    $\nu \langle u_i \nabla^2 u_i \rangle$, the Nyquist modes included, and equals
    $2 \nu \langle S_{ij} S_{ij} \rangle$ for a divergence-free velocity whose
    Nyquist modes vanish, since the strain rate leaves them out.
    """
    return _mean_square(mesh, velocity_spectral, 1, mesh.wavenumbers_squared).mul_(
        viscosity
    )


def enstrophy(mesh: Mesh, velocity_spectral: torch.Tensor) -> torch.Tensor:
    r"""Return the enstrophy $\tfrac12 \langle \omega^2 \rangle$, in 2D or 3D."""
    if mesh.dimension == 2:
        return _mean_square(mesh, mesh.curl_2d(velocity_spectral)).mul_(0.5)
    if mesh.dimension == 3:
        return _mean_square(mesh, mesh.curl_3d(velocity_spectral), 1).mul_(0.5)
    raise ValueError(f"enstrophy needs a 2D or 3D mesh, got {mesh.dimension}D")


def divergence_rms(mesh: Mesh, velocity_spectral: torch.Tensor) -> torch.Tensor:
    """Return the root mean square of the divergence."""
    return _mean_square(mesh, mesh.divergence(velocity_spectral)).sqrt_()


def relative_error(
    mesh: Mesh,
    field_spectral: torch.Tensor,
    reference_spectral: torch.Tensor,
    component_axes: int = 0,
) -> torch.Tensor:
    r"""Return the relative L2 error $\|f - r\| / \|r\|$ of a field against a reference.

    ``component_axes`` is the number of component axes, 0 for a scalar and 1 for a
    vector.
    """
    difference = _mean_square(mesh, field_spectral - reference_spectral, component_axes)
    reference = _mean_square(mesh, reference_spectral, component_axes)
    return difference.div_(reference).sqrt_()


def longitudinal_derivatives(
    mesh: Mesh, velocity_spectral: torch.Tensor
) -> torch.Tensor:
    r"""Return the longitudinal derivatives $\partial_i u_i$ at the mesh points.

    There is no sum over $i$: the component axis holds one derivative per component.
    """
    derivatives_spectral = velocity_spectral * mesh.derivative_wavenumbers
    return mesh.decomposition.to_physical(derivatives_spectral.mul_(1j))


def transverse_derivatives(mesh: Mesh, velocity_spectral: torch.Tensor) -> torch.Tensor:
    r"""Return the transverse derivatives $\partial_j u_i$ at the mesh points.

    The pairs $i \ne j$ lie on one component axis of size $d (d - 1)$, in row-major
    order of $(i, j)$.
    """
    d = mesh.dimension
    gradient_spectral = mesh.gradient(velocity_spectral).flatten(-d - 2, -d - 1)
    pairs = [i * d + j for i in range(d) for j in range(d) if i != j]
    index = torch.tensor(pairs, device=gradient_spectral.device)
    off_diagonal_spectral = gradient_spectral.index_select(-d - 1, index)
    return mesh.decomposition.to_physical(off_diagonal_spectral)


def local_dissipation(
    mesh: Mesh, velocity_spectral: torch.Tensor, viscosity: float
) -> torch.Tensor:
    r"""Return the local dissipation $2 \nu S_{ij} S_{ij}$ at the mesh points.

    $S_{ij} = \tfrac12 (\partial_j u_i + \partial_i u_j)$ is the strain rate.
    """
    d = mesh.dimension
    gradient_physical = mesh.decomposition.to_physical(mesh.gradient(velocity_spectral))
    component_dims = (-d - 2, -d - 1)
    twice_strain_physical = gradient_physical + gradient_physical.transpose(
        *component_dims
    )
    return twice_strain_physical.square_().sum(dim=component_dims).mul_(0.5 * viscosity)


def local_enstrophy(mesh: Mesh, velocity_spectral: torch.Tensor) -> torch.Tensor:
    r"""Return the local enstrophy $\tfrac12 \omega^2$ at the mesh points, in 2D or 3D.

    It raises a ``ValueError`` on a 1D mesh.
    """
    if mesh.dimension == 2:
        vorticity_spectral = mesh.curl_2d(velocity_spectral)
        vorticity_physical = mesh.decomposition.to_physical(vorticity_spectral)
        return vorticity_physical.square_().mul_(0.5)
    if mesh.dimension == 3:
        vorticity_spectral = mesh.curl_3d(velocity_spectral)
        vorticity_physical = mesh.decomposition.to_physical(vorticity_spectral)
        return vorticity_physical.square_().sum(dim=-4).mul_(0.5)
    raise ValueError(f"local enstrophy needs a 2D or 3D mesh, got {mesh.dimension}D")

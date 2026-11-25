r"""Spectra of spectral fields, binned into shells of integer wavenumbers.

Shell $m$ holds the wavenumbers with $\mathrm{round}(|n|) = m$, and shell 0 holds the
mean alone. A spectrum is a density in $\kappa$ on a last axis of shells, so that its
sum over the shells times $\Delta\kappa$ is the mean it splits; it keeps the leading
axes.
"""

import math
from dataclasses import dataclass, field

import torch

from specflow.mesh import Mesh


@dataclass(frozen=True, eq=False)
class Shells:
    r"""The shells of a mesh, up to its corners.

    There are $\mathrm{round}(\sqrt{d} \, N / 2) + 1$ shells of width
    $\Delta\kappa = 2 \pi / L$. ``indices`` holds the shell of each local wavenumber,
    ``wavenumbers`` the centres $\kappa_m = m \, \Delta\kappa$, and ``counts`` the
    number of modes in each shell over the full set of wavenumbers.
    """

    mesh: Mesh
    width: float = field(init=False)
    indices: torch.Tensor = field(init=False, repr=False)
    wavenumbers: torch.Tensor = field(init=False, repr=False)
    counts: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mesh = self.mesh
        width = 2 * math.pi / mesh.length
        count = round(math.sqrt(mesh.dimension) * mesh.points / 2) + 1
        magnitudes = mesh.wavenumbers_squared.div(width**2).sqrt_()
        indices = magnitudes.round_().to(torch.int64)
        wavenumbers = torch.arange(count, dtype=mesh.dtype, device=mesh.device)
        modes = mesh.parseval_weights.to(torch.int64)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "wavenumbers", wavenumbers.mul_(width))
        object.__setattr__(
            self, "counts", mesh.decomposition.binned_sum(modes, indices, count)
        )


def energy_spectrum(shells: Shells, velocity_spectral: torch.Tensor) -> torch.Tensor:
    r"""Return the energy spectrum $E(\kappa_m)$, of $\tfrac12 |\hat u_i|^2$."""
    mesh = shells.mesh
    density_spectral = velocity_spectral.abs().square_().sum(dim=-mesh.dimension - 1)
    return _shell_sum(shells, density_spectral.mul_(mesh.parseval_weights).mul_(0.5))


def budget_spectrum(
    shells: Shells, velocity_spectral: torch.Tensor, contribution_spectral: torch.Tensor
) -> torch.Tensor:
    r"""Return the budget spectrum $B(\kappa_m)$ of a contribution $c$.

    It splits $\mathrm{Re}(\hat u_i^* \hat c_i)$, so that its sum over the shells times
    $\Delta\kappa$ is the contribution's share of the rate of change of the kinetic
    energy.
    """
    mesh = shells.mesh
    product_spectral = torch.mul(velocity_spectral.conj(), contribution_spectral)
    density_spectral = product_spectral.real.sum(dim=-mesh.dimension - 1)
    return _shell_sum(shells, density_spectral.mul_(mesh.parseval_weights))


def flux(shells: Shells, budget: torch.Tensor) -> torch.Tensor:
    r"""Return the flux $\Pi(\kappa_m) = -\sum_{m' \le m} B(\kappa_{m'})\,\Delta\kappa$.

    The value of shell $m$ belongs to its upper edge $(m + \tfrac12) \, \Delta\kappa$,
    and is positive outward.
    """
    return budget.cumsum(dim=-1).mul_(-shells.width)


def _shell_sum(shells: Shells, density_spectral: torch.Tensor) -> torch.Tensor:
    totals = shells.mesh.decomposition.binned_sum(
        density_spectral, shells.indices, len(shells.wavenumbers)
    )
    return totals.div_(shells.width)

r"""Turbulence scales from global diagnostics, with isotropic 3D formulas.

The scales are functions of tensors, such as the kinetic energy of the fluctuation
$k$, the dissipation $\varepsilon$ and the viscosity $\nu$, and broadcast over their
leading axes. They use $u_\text{rms} = \sqrt{2k/3}$, and a scale that divides by the
dissipation is infinite where the dissipation is not positive.
"""

import math

import torch

from specflow.mesh import Mesh


def rms_velocity(fluctuation_kinetic_energy: torch.Tensor) -> torch.Tensor:
    r"""Return the rms velocity $u_\text{rms} = \sqrt{2k/3}$."""
    return fluctuation_kinetic_energy.mul(2 / 3).sqrt_()


def taylor_microscale(
    fluctuation_kinetic_energy: torch.Tensor,
    dissipation: torch.Tensor,
    viscosity: torch.Tensor | float,
) -> torch.Tensor:
    r"""Return the Taylor microscale.

    It is $\lambda = \sqrt{15 \nu u_\text{rms}^2 / \varepsilon}$.
    """
    ratio = viscosity / dissipation
    squared = ratio * rms_velocity(fluctuation_kinetic_energy).square_()
    return _dissipative(dissipation, squared.mul_(15).sqrt_())


def taylor_reynolds_number(
    fluctuation_kinetic_energy: torch.Tensor,
    dissipation: torch.Tensor,
    viscosity: torch.Tensor | float,
) -> torch.Tensor:
    r"""Return the Reynolds number $Re_\lambda = u_\text{rms} \lambda / \nu$."""
    microscale = taylor_microscale(fluctuation_kinetic_energy, dissipation, viscosity)
    product = microscale.mul_(rms_velocity(fluctuation_kinetic_energy))
    return _dissipative(dissipation, product.div_(viscosity))


def kolmogorov_length(
    dissipation: torch.Tensor, viscosity: torch.Tensor | float
) -> torch.Tensor:
    r"""Return the Kolmogorov length $\eta = (\nu^3 / \varepsilon)^{1/4}$."""
    return _dissipative(dissipation, (viscosity**3 / dissipation).pow_(0.25))


def kolmogorov_time(
    dissipation: torch.Tensor, viscosity: torch.Tensor | float
) -> torch.Tensor:
    r"""Return the Kolmogorov time $\tau_\eta = (\nu / \varepsilon)^{1/2}$."""
    return _dissipative(dissipation, (viscosity / dissipation).sqrt_())


def integral_length(spectrum: torch.Tensor, wavenumbers: torch.Tensor) -> torch.Tensor:
    r"""Return the integral length of an energy spectrum on its shell centres.

    It is $L_u = \frac{\pi}{2 u_\text{rms}^2} \sum_{m \ge 1} E(\kappa_m) / \kappa_m \,
    \Delta\kappa$, with $u_\text{rms}$ from the same shells, so that the mean in shell 0
    is left out.
    """
    width = wavenumbers[1]
    fluctuation = spectrum[..., 1:]
    weighted = (fluctuation / wavenumbers[1:]).sum(-1).mul_(width)
    squared = rms_velocity(_shell_energy(fluctuation, width)).square_()
    return weighted.div_(squared).mul_(math.pi / 2)


def large_eddy_time(spectrum: torch.Tensor, wavenumbers: torch.Tensor) -> torch.Tensor:
    r"""Return the large-eddy time $T_L = L_u / u_\text{rms}$ of an energy spectrum.

    $u_\text{rms}$ comes from the shells $m \ge 1$, as in the integral length.
    """
    energy = _shell_energy(spectrum[..., 1:], wavenumbers[1])
    return integral_length(spectrum, wavenumbers).div_(rms_velocity(energy))


def resolution(
    mesh: Mesh, dissipation: torch.Tensor, viscosity: torch.Tensor | float
) -> torch.Tensor:
    r"""Return the resolution $\kappa_\text{max} \eta$ of a mesh.

    $\kappa_\text{max} = \frac{2 \pi}{L} \frac{N}{3}$ is the largest wavenumber the
    spherical 2/3 truncation keeps.
    """
    largest = 2 * math.pi / mesh.length * mesh.points / 3
    return kolmogorov_length(dissipation, viscosity).mul_(largest)


def _shell_energy(spectrum: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    return spectrum.sum(-1).mul_(width)


def _dissipative(dissipation: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return scale.masked_fill_(dissipation <= 0, math.inf)

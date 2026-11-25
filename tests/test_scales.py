import math
import warnings

import pytest
import torch

from specflow import diagnostics, scales
from specflow.decomposition import SerialDecomposition
from specflow.mesh import Mesh
from specflow.spectra import Shells, energy_spectrum

POINTS = 12
VISCOSITY = 0.01
TOLERANCE = 1e-14


def make_mesh() -> Mesh:
    return Mesh(POINTS, 2 * math.pi, 3, SerialDecomposition(POINTS, 3))


def taylor_green_spectral(mesh: Mesh) -> torch.Tensor:
    x, y, z = mesh.coordinates
    components = [
        torch.sin(x) * torch.cos(y) * torch.cos(z),
        -torch.cos(x) * torch.sin(y) * torch.cos(z),
        torch.zeros_like(x),
    ]
    return mesh.decomposition.to_spectral(torch.stack(components))


def taylor_green_scalars(mesh: Mesh) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = taylor_green_spectral(mesh)
    energy = diagnostics.fluctuation_kinetic_energy(mesh, velocity)
    return energy, diagnostics.dissipation(mesh, velocity, VISCOSITY)


def assert_matches(actual: torch.Tensor, expected: float) -> None:
    assert actual.item() == pytest.approx(expected, rel=TOLERANCE)


def test_the_point_scales_match_the_taylor_green_vortex() -> None:
    mesh = make_mesh()
    energy, dissipation = taylor_green_scalars(mesh)

    rms = scales.rms_velocity(energy)
    microscale = scales.taylor_microscale(energy, dissipation, VISCOSITY)
    reynolds = scales.taylor_reynolds_number(energy, dissipation, VISCOSITY)
    length = scales.kolmogorov_length(dissipation, VISCOSITY)
    time = scales.kolmogorov_time(dissipation, VISCOSITY)
    resolution = scales.resolution(mesh, dissipation, VISCOSITY)

    kolmogorov = (4 * VISCOSITY**2 / 3) ** 0.25
    assert_matches(rms, math.sqrt(1 / 12))
    assert_matches(microscale, math.sqrt(5 / 3))
    assert_matches(reynolds, math.sqrt(5) / (6 * VISCOSITY))
    assert_matches(length, kolmogorov)
    assert_matches(time, math.sqrt(4 / 3))
    assert_matches(resolution, POINTS / 3 * kolmogorov)


def test_the_large_scales_match_the_taylor_green_vortex() -> None:
    mesh = make_mesh()
    shells = Shells(mesh)
    spectrum = energy_spectrum(shells, taylor_green_spectral(mesh))

    length = scales.integral_length(spectrum, shells.wavenumbers)
    time = scales.large_eddy_time(spectrum, shells.wavenumbers)

    assert_matches(length, 3 * math.pi / 8)
    assert_matches(time, 3 * math.sqrt(3) * math.pi / 4)


def test_the_large_scales_do_not_depend_on_the_domain_length_in_its_units() -> None:
    length = 3.0
    mesh = Mesh(POINTS, length, 3, SerialDecomposition(POINTS, 3))
    shells = Shells(mesh)
    x, y, z = mesh.coordinates.mul(2 * math.pi / length)
    velocity_physical = torch.stack(
        [
            torch.sin(x) * torch.cos(y) * torch.cos(z),
            -torch.cos(x) * torch.sin(y) * torch.cos(z),
            torch.zeros_like(x),
        ]
    )
    spectrum = energy_spectrum(
        shells, mesh.decomposition.to_spectral(velocity_physical)
    )

    integral = scales.integral_length(spectrum, shells.wavenumbers)
    time = scales.large_eddy_time(spectrum, shells.wavenumbers)

    unit = length / (2 * math.pi)
    assert_matches(integral, 3 * math.pi / 8 * unit)
    assert_matches(time, 3 * math.sqrt(3) * math.pi / 4 * unit)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_dissipation_that_is_not_positive_gives_infinite_scales(
    value: float,
) -> None:
    mesh = make_mesh()
    energies = torch.tensor([0.0, 0.125], dtype=mesh.dtype)
    dissipation = torch.full((2,), value, dtype=mesh.dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = [
            scales.taylor_microscale(energies, dissipation, VISCOSITY),
            scales.taylor_reynolds_number(energies, dissipation, VISCOSITY),
            scales.kolmogorov_length(dissipation, VISCOSITY),
            scales.kolmogorov_time(dissipation, VISCOSITY),
            scales.resolution(mesh, dissipation, VISCOSITY),
        ]

    for result in results:
        assert torch.equal(result, torch.full((2,), math.inf, dtype=mesh.dtype))


def test_the_point_scales_broadcast_over_the_leading_axes() -> None:
    mesh = make_mesh()
    energies = torch.tensor([[0.125], [0.5]], dtype=mesh.dtype)
    dissipations = torch.tensor([0.0075, 0.02, 0.1], dtype=mesh.dtype)
    viscosities = torch.tensor([[[0.01]], [[0.02]]], dtype=mesh.dtype)

    microscale = scales.taylor_microscale(energies, dissipations, viscosities)
    reynolds = scales.taylor_reynolds_number(energies, dissipations, viscosities)
    length = scales.kolmogorov_length(dissipations, viscosities)
    time = scales.kolmogorov_time(dissipations, viscosities)
    resolution = scales.resolution(mesh, dissipations, viscosities)

    assert microscale.shape == (2, 2, 3)
    assert reynolds.shape == (2, 2, 3)
    assert length.shape == (2, 1, 3)
    assert time.shape == (2, 1, 3)
    assert resolution.shape == (2, 1, 3)
    for n, viscosity in enumerate(viscosities.flatten()):
        for i, energy in enumerate(energies[:, 0]):
            for j, dissipation in enumerate(dissipations):
                single = scales.taylor_reynolds_number(energy, dissipation, viscosity)
                assert reynolds[n, i, j].item() == pytest.approx(single.item())
        for j, dissipation in enumerate(dissipations):
            single = scales.kolmogorov_length(dissipation, viscosity)
            assert length[n, 0, j].item() == pytest.approx(single.item())


def test_the_large_scales_broadcast_over_the_leading_axes() -> None:
    mesh = make_mesh()
    shells = Shells(mesh)
    generator = torch.Generator().manual_seed(0)
    count = len(shells.wavenumbers)
    spectra = torch.rand((2, 3, count), generator=generator, dtype=mesh.dtype)

    lengths = scales.integral_length(spectra, shells.wavenumbers)
    times = scales.large_eddy_time(spectra, shells.wavenumbers)

    assert lengths.shape == (2, 3)
    assert times.shape == (2, 3)
    assert lengths.dtype == mesh.dtype
    for i in range(2):
        for j in range(3):
            length = scales.integral_length(spectra[i, j], shells.wavenumbers)
            time = scales.large_eddy_time(spectra[i, j], shells.wavenumbers)
            assert lengths[i, j].item() == pytest.approx(length.item(), rel=TOLERANCE)
            assert times[i, j].item() == pytest.approx(time.item(), rel=TOLERANCE)


def test_the_integral_length_leaves_out_the_mean() -> None:
    mesh = make_mesh()
    shells = Shells(mesh)
    spectrum = energy_spectrum(shells, taylor_green_spectral(mesh))
    with_mean = spectrum.clone()
    with_mean[0] = 10.0

    length = scales.integral_length(with_mean, shells.wavenumbers)

    assert_matches(length, 3 * math.pi / 8)

import math

import pytest
import torch

from specflow import diagnostics
from specflow.decomposition import SerialDecomposition
from specflow.mesh import Mesh
from specflow.spectra import Shells, budget_spectrum

POINTS = 8
VISCOSITY = 1 / 1600


def make_mesh(dimension: int) -> Mesh:
    return Mesh(POINTS, 2 * math.pi, dimension, SerialDecomposition(POINTS, dimension))


def taylor_green_spectral(mesh: Mesh) -> torch.Tensor:
    coordinates = mesh.coordinates
    x, y = coordinates[0], coordinates[1]
    shape = torch.ones_like(x)
    if mesh.dimension == 3:
        shape = torch.cos(coordinates[2])
    components = [
        torch.sin(x) * torch.cos(y) * shape,
        -torch.cos(x) * torch.sin(y) * shape,
    ]
    if mesh.dimension == 3:
        components.append(torch.zeros_like(x))
    return mesh.decomposition.to_spectral(torch.stack(components))


def test_the_3d_taylor_green_vortex_matches_the_reference_first_row() -> None:
    mesh = make_mesh(3)
    velocity_spectral = taylor_green_spectral(mesh)

    energy = diagnostics.kinetic_energy(mesh, velocity_spectral)
    dissipation = diagnostics.dissipation(mesh, velocity_spectral, VISCOSITY)
    enstrophy = diagnostics.enstrophy(mesh, velocity_spectral)

    assert energy.item() == pytest.approx(0.125, rel=1e-14)
    assert dissipation.item() == pytest.approx(0.75 * VISCOSITY, rel=1e-14)
    assert enstrophy.item() == pytest.approx(0.375, rel=1e-14)


def test_the_2d_taylor_green_vortex_has_known_energy_and_enstrophy() -> None:
    mesh = make_mesh(2)
    velocity_spectral = taylor_green_spectral(mesh)

    energy = diagnostics.kinetic_energy(mesh, velocity_spectral)
    dissipation = diagnostics.dissipation(mesh, velocity_spectral, VISCOSITY)
    enstrophy = diagnostics.enstrophy(mesh, velocity_spectral)

    assert energy.item() == pytest.approx(0.25, rel=1e-14)
    assert dissipation.item() == pytest.approx(VISCOSITY, rel=1e-14)
    assert enstrophy.item() == pytest.approx(0.5, rel=1e-14)


def test_the_enstrophy_of_a_1d_mesh_raises() -> None:
    mesh = make_mesh(1)

    with pytest.raises(ValueError, match="1D"):
        diagnostics.enstrophy(mesh, torch.zeros(1, 5, dtype=mesh.complex_dtype))


def test_a_divergence_free_field_has_zero_rms_divergence() -> None:
    mesh = make_mesh(3)

    rms = diagnostics.divergence_rms(mesh, taylor_green_spectral(mesh))

    assert rms.item() <= 1e-15


def test_the_rms_divergence_of_a_gradient_is_known() -> None:
    mesh = make_mesh(2)
    x = mesh.coordinates[0]
    velocity_physical = torch.stack([-torch.sin(x), torch.zeros_like(x)])

    rms = diagnostics.divergence_rms(
        mesh, mesh.decomposition.to_spectral(velocity_physical)
    )

    assert rms.item() == pytest.approx(1 / math.sqrt(2), rel=1e-14)


@pytest.mark.parametrize("component_axes", [0, 1])
def test_the_relative_error_of_a_scaled_field_is_the_scale_minus_one(
    component_axes: int,
) -> None:
    mesh = make_mesh(2)
    reference_spectral = taylor_green_spectral(mesh)
    if component_axes == 0:
        reference_spectral = reference_spectral[0]

    error = diagnostics.relative_error(
        mesh, 1.1 * reference_spectral, reference_spectral, component_axes
    )

    assert error.shape == ()
    assert error.item() == pytest.approx(0.1, rel=1e-12)


def test_diagnostics_keep_the_leading_axes_on_the_device() -> None:
    mesh = make_mesh(2)
    velocity_spectral = taylor_green_spectral(mesh)
    ensemble_spectral = torch.stack([velocity_spectral, 2 * velocity_spectral])

    energy = diagnostics.kinetic_energy(mesh, ensemble_spectral)

    assert energy.shape == (2,)
    assert energy.dtype == mesh.dtype
    assert energy.device == mesh.wavenumbers.device
    assert energy.tolist() == pytest.approx([0.25, 1.0], rel=1e-14)


@pytest.mark.parametrize("dimension", [2, 3])
def test_the_fluctuation_kinetic_energy_ignores_a_uniform_mean(dimension: int) -> None:
    mesh = make_mesh(dimension)
    velocity_spectral = taylor_green_spectral(mesh)
    shifted_spectral = velocity_spectral.clone()
    shifted_spectral[(..., *mesh.dimension * (0,))] += 0.3

    fluctuation = diagnostics.fluctuation_kinetic_energy(mesh, shifted_spectral)

    expected = diagnostics.kinetic_energy(mesh, velocity_spectral)
    assert fluctuation.item() == pytest.approx(expected.item(), rel=1e-14)
    assert diagnostics.kinetic_energy(mesh, shifted_spectral) > fluctuation


def random_spectral(mesh: Mesh, *leading: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (*leading, mesh.dimension, *mesh.physical_shape)
    field_physical = torch.randn(shape, generator=generator, dtype=mesh.dtype)
    return mesh.project(mesh.decomposition.to_spectral(field_physical))


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_the_injection_rate_is_the_sum_of_the_forcing_budget_spectrum(
    dimension: int,
) -> None:
    mesh = Mesh(POINTS, 3.0, dimension, SerialDecomposition(POINTS, dimension))
    shells = Shells(mesh)
    velocity_spectral = random_spectral(mesh, 2, seed=0)
    forcing_spectral = random_spectral(mesh, 2, seed=1)

    rate = diagnostics.injection_rate(mesh, velocity_spectral, forcing_spectral)

    budget = budget_spectrum(shells, velocity_spectral, forcing_spectral)
    expected = budget.sum(-1) * shells.width
    assert rate.shape == (2,)
    assert rate.dtype == mesh.dtype
    assert rate.device == mesh.wavenumbers.device
    torch.testing.assert_close(rate, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("phase", [0.0, 1.0, math.pi / 2, math.pi])
def test_the_injection_rate_of_single_modes_is_known(phase: float) -> None:
    mesh = make_mesh(3)
    z = mesh.coordinates[2]
    zero = torch.zeros_like(z)
    velocity_physical = torch.stack([1.5 * torch.cos(2 * z), zero, zero])
    forcing_physical = torch.stack([0.4 * torch.cos(2 * z + phase), zero, zero])

    rate = diagnostics.injection_rate(
        mesh,
        mesh.decomposition.to_spectral(velocity_physical),
        mesh.decomposition.to_spectral(forcing_physical),
    )

    expected = 0.5 * 1.5 * 0.4 * math.cos(phase)
    assert rate.item() == pytest.approx(expected, rel=1e-14, abs=1e-15)

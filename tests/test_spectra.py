import math

import pytest
import torch

from specflow import diagnostics
from specflow.dealiasing import TwoThirdsDealiasing
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.mesh import Mesh
from specflow.spectra import Shells, budget_spectrum, energy_spectrum, flux
from specflow.terms import Advection, Diffusion

POINTS = 12
TOLERANCE = 1e-14
DIMENSIONS = [1, 2, 3]
LENGTHS = [2 * math.pi, 3.0]
VISCOSITY = 0.1


def make_mesh(dimension: int, length: float = 2 * math.pi) -> Mesh:
    decomposition = SerialDecomposition(POINTS, dimension)
    return Mesh(POINTS, length, dimension, decomposition)


def velocity_spectral(mesh: Mesh, *leading: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (*leading, mesh.dimension, *mesh.physical_shape)
    field_physical = torch.randn(shape, generator=generator, dtype=mesh.dtype)
    field_spectral = mesh.decomposition.to_spectral(field_physical)
    integers_squared = mesh.wavenumbers_squared * (mesh.length / (2 * math.pi)) ** 2
    field_spectral.mul_(integers_squared < (POINTS / 3) ** 2)
    return mesh.project(field_spectral)


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


def shell_width(mesh: Mesh) -> float:
    return 2 * math.pi / mesh.length


def navier_stokes(mesh: Mesh) -> Equation:
    terms = [
        Diffusion(mesh, "velocity", VISCOSITY, components=(mesh.dimension,)),
        Advection(mesh, TwoThirdsDealiasing(mesh), "velocity"),
    ]
    return Equation("velocity", terms, projection=mesh.project)


def tendency_spectral(
    equation: Equation, velocity_spectral: torch.Tensor
) -> torch.Tensor:
    nonlinear_spectral = torch.full_like(velocity_spectral, math.nan)
    equation.nonlinear_tendency(
        {"velocity": velocity_spectral}, 0.0, nonlinear_spectral
    )
    return nonlinear_spectral.add_(equation.linear_operator * velocity_spectral)


def energy_rate(
    mesh: Mesh, velocity_spectral: torch.Tensor, tendency_spectral: torch.Tensor
) -> torch.Tensor:
    forward = diagnostics.kinetic_energy(mesh, velocity_spectral + tendency_spectral)
    backward = diagnostics.kinetic_energy(mesh, velocity_spectral - tendency_spectral)
    return (forward - backward) / 2


def budget_spectra(
    mesh: Mesh, velocity_spectral: torch.Tensor
) -> dict[str, torch.Tensor]:
    shells = Shells(mesh)
    contributions = navier_stokes(mesh).contributions(
        {"velocity": velocity_spectral}, 0.0
    )
    return {
        name: budget_spectrum(shells, velocity_spectral, contribution_spectral)
        for name, contribution_spectral in contributions.items()
    }


def budget_velocity(dimension: int, fixture: str) -> tuple[Mesh, torch.Tensor]:
    mesh = make_mesh(dimension)
    if fixture == "taylor_green":
        return mesh, taylor_green_spectral(mesh)
    return mesh, velocity_spectral(mesh)


def assert_budget_close(
    actual: torch.Tensor, expected: torch.Tensor, scale: torch.Tensor
) -> None:
    atol = TOLERANCE * max(float(scale.abs().max().item()), 1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=atol)


def assert_sums_to_the_energies(mesh: Mesh, velocity_spectral: torch.Tensor) -> None:
    shells = Shells(mesh)

    spectrum = energy_spectrum(shells, velocity_spectral)

    energy = diagnostics.kinetic_energy(mesh, velocity_spectral)
    fluctuation = diagnostics.fluctuation_kinetic_energy(mesh, velocity_spectral)
    total = spectrum.sum(-1) * shell_width(mesh)
    without_mean = spectrum[..., 1:].sum(-1) * shell_width(mesh)
    torch.testing.assert_close(total, energy, rtol=TOLERANCE, atol=0)
    torch.testing.assert_close(without_mean, fluctuation, rtol=TOLERANCE, atol=0)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_there_is_one_shell_up_to_the_corners_of_the_mesh(dimension: int) -> None:
    mesh = make_mesh(dimension)

    shells = Shells(mesh)

    count = round(math.sqrt(dimension) * POINTS / 2) + 1
    assert shells.wavenumbers.shape == (count,)
    assert shells.counts.shape == (count,)
    assert int(shells.indices.max().item()) == count - 1


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_the_mode_counts_total_the_number_of_wavenumbers(dimension: int) -> None:
    mesh = make_mesh(dimension)

    shells = Shells(mesh)

    assert int(shells.counts.sum().item()) == POINTS**dimension
    assert int(shells.counts[0].item()) == 1


def test_the_mode_counts_match_a_count_of_the_integer_wavenumbers() -> None:
    mesh = make_mesh(2)
    integers = torch.arange(POINTS) - POINTS // 2
    magnitudes = (integers[:, None] ** 2 + integers[None, :] ** 2).double().sqrt()

    shells = Shells(mesh)

    expected = torch.bincount(magnitudes.round().long().flatten())
    assert torch.equal(shells.counts, expected)


@pytest.mark.parametrize("length", LENGTHS)
def test_the_shell_centres_are_multiples_of_the_shell_width(length: float) -> None:
    mesh = make_mesh(3, length)

    shells = Shells(mesh)

    expected = torch.arange(len(shells.wavenumbers), dtype=mesh.dtype)
    expected.mul_(shell_width(mesh))
    assert shells.wavenumbers.dtype == mesh.dtype
    assert shells.wavenumbers.device == mesh.device
    torch.testing.assert_close(shells.wavenumbers, expected, rtol=TOLERANCE, atol=0)


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_the_spectrum_of_a_random_velocity_sums_to_its_energies(
    dimension: int, length: float
) -> None:
    mesh = make_mesh(dimension, length)
    velocity = velocity_spectral(mesh)

    assert diagnostics.fluctuation_kinetic_energy(mesh, velocity) < (
        diagnostics.kinetic_energy(mesh, velocity)
    )
    assert_sums_to_the_energies(mesh, velocity)


@pytest.mark.parametrize("dimension", [2, 3])
def test_the_spectrum_of_the_taylor_green_vortex_sums_to_its_energies(
    dimension: int,
) -> None:
    mesh = make_mesh(dimension)

    assert_sums_to_the_energies(mesh, taylor_green_spectral(mesh))


@pytest.mark.parametrize("dimension", [2, 3])
def test_the_taylor_green_vortex_lies_in_one_shell(dimension: int) -> None:
    mesh = make_mesh(dimension)
    shell = round(math.sqrt(dimension))

    spectrum = energy_spectrum(Shells(mesh), taylor_green_spectral(mesh))

    energy = 0.25 if dimension == 2 else 0.125
    expected = torch.zeros_like(spectrum)
    expected[shell] = energy / shell_width(mesh)
    torch.testing.assert_close(spectrum, expected, rtol=TOLERANCE, atol=TOLERANCE)


def test_the_energy_spectrum_keeps_the_leading_axes_on_the_device() -> None:
    mesh = make_mesh(3)
    shells = Shells(mesh)
    ensemble = velocity_spectral(mesh, 2, 3)

    spectrum = energy_spectrum(shells, ensemble)

    assert spectrum.shape == (2, 3, len(shells.wavenumbers))
    assert spectrum.dtype == mesh.dtype
    assert spectrum.device == mesh.device
    for index in range(2):
        for other in range(3):
            single = energy_spectrum(shells, ensemble[index, other])
            torch.testing.assert_close(
                spectrum[index, other], single, rtol=TOLERANCE, atol=0
            )


@pytest.mark.parametrize("fixture", ["random", "taylor_green"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_the_advection_budget_spectrum_sums_to_zero(
    dimension: int, fixture: str
) -> None:
    mesh, velocity = budget_velocity(dimension, fixture)

    advection = budget_spectra(mesh, velocity)["advection"]

    total = advection.sum(-1) * shell_width(mesh)
    assert_budget_close(total, torch.zeros_like(total), advection.abs().sum())


@pytest.mark.parametrize("fixture", ["random", "taylor_green"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_the_diffusion_budget_spectrum_sums_to_minus_the_dissipation(
    dimension: int, fixture: str
) -> None:
    mesh, velocity = budget_velocity(dimension, fixture)

    diffusion = budget_spectra(mesh, velocity)["diffusion"]

    dissipation = diagnostics.dissipation(mesh, velocity, VISCOSITY)
    total = diffusion.sum(-1) * shell_width(mesh)
    assert_budget_close(total, -dissipation, dissipation)


@pytest.mark.parametrize("fixture", ["random", "taylor_green"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_the_budget_spectra_sum_to_the_rate_of_change_of_the_energy(
    dimension: int, fixture: str
) -> None:
    mesh, velocity = budget_velocity(dimension, fixture)
    tendency = tendency_spectral(navier_stokes(mesh), velocity)

    spectra = budget_spectra(mesh, velocity)

    rate = energy_rate(mesh, velocity, tendency)
    total = sum(spectra.values(), torch.zeros(())).sum(-1) * shell_width(mesh)
    assert_budget_close(total, rate, rate)


@pytest.mark.parametrize("fixture", ["random", "taylor_green"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_the_flux_of_advection_vanishes_past_the_last_shell(
    dimension: int, fixture: str
) -> None:
    mesh, velocity = budget_velocity(dimension, fixture)
    advection = budget_spectra(mesh, velocity)["advection"]

    pi = flux(Shells(mesh), advection)

    assert_budget_close(pi[-1], torch.zeros_like(pi[-1]), advection.abs().sum())


@pytest.mark.parametrize("length", LENGTHS)
def test_the_flux_is_minus_the_cumulative_budget_spectrum(length: float) -> None:
    mesh = make_mesh(3, length)
    shells = Shells(mesh)
    budget = torch.randn(2, len(shells.wavenumbers), dtype=mesh.dtype)

    pi = flux(shells, budget)

    expected = torch.zeros_like(budget)
    for shell in range(budget.shape[-1]):
        expected[:, shell] = -budget[:, : shell + 1].sum(-1) * shell_width(mesh)
    assert pi.dtype == budget.dtype
    torch.testing.assert_close(pi, expected, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.mark.parametrize("dimension", [2, 3])
def test_advection_moves_the_energy_of_a_random_velocity_between_shells(
    dimension: int,
) -> None:
    mesh = make_mesh(dimension)
    velocity = velocity_spectral(mesh)
    advection = budget_spectra(mesh, velocity)["advection"]

    pi = flux(Shells(mesh), advection)

    assert advection.abs().max() > 1e3 * TOLERANCE
    assert pi.abs().max() > 1e3 * TOLERANCE


def test_the_budget_spectrum_keeps_the_leading_axes_on_the_device() -> None:
    mesh = make_mesh(3)
    shells = Shells(mesh)
    ensemble = velocity_spectral(mesh, 2)
    contributions = navier_stokes(mesh).contributions({"velocity": ensemble}, 0.0)
    advection_spectral = contributions["advection"]

    budget = budget_spectrum(shells, ensemble, advection_spectral)

    assert budget.shape == (2, len(shells.wavenumbers))
    assert budget.dtype == mesh.dtype
    assert budget.device == mesh.device
    for index in range(2):
        single = budget_spectrum(shells, ensemble[index], advection_spectral[index])
        torch.testing.assert_close(budget[index], single, rtol=TOLERANCE, atol=0)

import math

import pytest
import torch

from specflow import diagnostics
from specflow.decomposition import SerialDecomposition
from specflow.mesh import Mesh
from specflow.spectra import Shells, energy_spectrum

POINTS = 12
TOLERANCE = 1e-14
DIMENSIONS = [1, 2, 3]
LENGTHS = [2 * math.pi, 3.0]


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

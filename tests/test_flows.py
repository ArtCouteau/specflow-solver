import math

import pytest
import torch

from specflow import flows
from specflow.decomposition import SerialDecomposition
from specflow.mesh import Mesh
from specflow.spectra import Shells, energy_spectrum

POINTS = 12
TOLERANCE = 1e-14
DIMENSIONS = [2, 3]
LENGTHS = [2 * math.pi, 3.0]


def make_mesh(dimension: int, length: float = 2 * math.pi) -> Mesh:
    return Mesh(POINTS, length, dimension, SerialDecomposition(POINTS, dimension))


def prescribed_spectrum(mesh: Mesh) -> torch.Tensor:
    shells = Shells(mesh)
    wavenumbers = shells.wavenumbers
    peak = 3 * shells.width
    spectrum = wavenumbers.pow(4).mul_(torch.exp(-2 * (wavenumbers / peak) ** 2))
    return spectrum.mul_(wavenumbers <= (POINTS // 2 - 1) * shells.width)


def random_velocity(mesh: Mesh, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return flows.random_velocity_spectral(mesh, prescribed_spectrum(mesh), generator)


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_the_energy_spectrum_equals_the_prescribed_one(
    dimension: int, length: float
) -> None:
    mesh = make_mesh(dimension, length)
    spectrum = prescribed_spectrum(mesh)

    velocity_spectral = random_velocity(mesh)

    actual = energy_spectrum(Shells(mesh), velocity_spectral)
    atol = TOLERANCE * float(spectrum.max().item())
    torch.testing.assert_close(actual, spectrum, rtol=TOLERANCE, atol=atol)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_the_velocity_is_divergence_free_with_no_mean(dimension: int) -> None:
    mesh = make_mesh(dimension)

    velocity_spectral = random_velocity(mesh)

    divergence_spectral = mesh.divergence(velocity_spectral)
    scale = float(velocity_spectral.abs().max().item())
    assert float(divergence_spectral.abs().max().item()) <= TOLERANCE * scale
    mean = velocity_spectral[(..., *dimension * (0,))]
    assert torch.equal(mean, torch.zeros_like(mean))


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_the_velocity_has_no_nyquist_mode(dimension: int) -> None:
    mesh = make_mesh(dimension)
    integers = mesh.wavenumbers.mul(mesh.length / (2 * math.pi)).round_()
    nyquist = (integers.abs() == POINTS // 2).any(dim=0)

    velocity_spectral = random_velocity(mesh)

    assert float(velocity_spectral[:, nyquist].abs().max().item()) == 0.0


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_the_velocity_is_real(dimension: int) -> None:
    mesh = make_mesh(dimension)

    velocity_spectral = random_velocity(mesh)

    velocity_physical = mesh.decomposition.to_physical(velocity_spectral)
    roundtrip_spectral = mesh.decomposition.to_spectral(velocity_physical)
    torch.testing.assert_close(
        roundtrip_spectral, velocity_spectral, rtol=0, atol=TOLERANCE
    )


def test_one_seed_gives_one_field() -> None:
    mesh = make_mesh(3)

    assert torch.equal(random_velocity(mesh, seed=1), random_velocity(mesh, seed=1))


def test_two_seeds_give_two_fields() -> None:
    mesh = make_mesh(3)

    assert not torch.equal(random_velocity(mesh, seed=1), random_velocity(mesh, seed=2))


def test_a_nonzero_value_on_the_mean_shell_raises() -> None:
    mesh = make_mesh(3)
    spectrum = prescribed_spectrum(mesh)
    spectrum[0] = 1.0

    with pytest.raises(ValueError, match="shell 0"):
        flows.random_velocity_spectral(mesh, spectrum, torch.Generator())


def test_a_positive_value_on_a_shell_without_energy_raises() -> None:
    mesh = make_mesh(3)
    spectrum = prescribed_spectrum(mesh)
    spectrum[-1] = 1.0

    with pytest.raises(ValueError, match="without energy"):
        flows.random_velocity_spectral(mesh, spectrum, torch.Generator())


def test_a_negative_value_raises() -> None:
    mesh = make_mesh(3)
    spectrum = prescribed_spectrum(mesh)
    spectrum[2] = -1.0

    with pytest.raises(ValueError, match="negative"):
        flows.random_velocity_spectral(mesh, spectrum, torch.Generator())


def test_a_spectrum_with_the_wrong_number_of_shells_raises() -> None:
    mesh = make_mesh(3)
    spectrum = prescribed_spectrum(mesh)[:-1]

    with pytest.raises(ValueError, match="shells"):
        flows.random_velocity_spectral(mesh, spectrum, torch.Generator())


@pytest.mark.gpu
def test_a_cpu_generator_gives_the_same_field_on_the_gpu() -> None:
    decomposition = SerialDecomposition(POINTS, 3)
    cpu_mesh = Mesh(POINTS, 2 * math.pi, 3, decomposition)
    gpu_mesh = Mesh(POINTS, 2 * math.pi, 3, decomposition, device=torch.device("cuda"))

    cpu_velocity_spectral = random_velocity(cpu_mesh)
    gpu_velocity_spectral = random_velocity(gpu_mesh)

    assert gpu_velocity_spectral.device.type == "cuda"
    atol = TOLERANCE * float(cpu_velocity_spectral.abs().max().item())
    torch.testing.assert_close(
        gpu_velocity_spectral.cpu(), cpu_velocity_spectral, rtol=0, atol=atol
    )

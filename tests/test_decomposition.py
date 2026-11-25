import math

import pytest
import torch

from specflow.decomposition import SerialDecomposition
from specflow.mesh import Mesh

DIMENSIONS = [1, 2, 3]


def make_mesh(points: int, dimension: int, length: float = 2 * math.pi) -> Mesh:
    return Mesh(points, length, dimension, SerialDecomposition(points, dimension))


def random_physical(mesh: Mesh, *leading: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    shape = (*leading, *(mesh.dimension * (mesh.points,)))
    return torch.randn(shape, generator=generator, dtype=torch.float64)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_round_trip_from_physical_space_is_exact(dimension: int) -> None:
    mesh = make_mesh(8, dimension)
    field_physical = random_physical(mesh, 2, 3)
    decomposition = mesh.decomposition
    round_trip = decomposition.to_physical(decomposition.to_spectral(field_physical))
    torch.testing.assert_close(round_trip, field_physical, rtol=0, atol=1e-14)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_spectral_and_physical_shapes_follow_the_slices(dimension: int) -> None:
    mesh = make_mesh(8, dimension)
    field_spectral = mesh.decomposition.to_spectral(random_physical(mesh, 4))
    field_physical = mesh.decomposition.to_physical(field_spectral)
    decomposition = mesh.decomposition
    spectral_lengths = tuple(s.stop - s.start for s in decomposition.spectral_slices)
    physical_lengths = tuple(s.stop - s.start for s in decomposition.physical_slices)
    assert spectral_lengths == (dimension - 1) * (8,) + (5,)
    assert physical_lengths == dimension * (8,)
    assert field_spectral.shape == (4, *spectral_lengths)
    assert field_physical.shape == (4, *physical_lengths)
    assert field_spectral.dtype == torch.complex128


def test_forward_normalization_gives_fourier_coefficients() -> None:
    mesh = make_mesh(16, 2, length=1.0)
    x, y = torch.meshgrid(
        torch.arange(16, dtype=torch.float64) / 16,
        torch.arange(16, dtype=torch.float64) / 16,
        indexing="ij",
    )
    field_physical = 3.0 + 2.0 * torch.cos(2 * math.pi * (x + 2 * y))
    field_spectral = mesh.decomposition.to_spectral(field_physical)
    expected = torch.zeros(16, 9, dtype=torch.complex128)
    expected[0, 0] = 3.0
    expected[1, 2] = 1.0
    torch.testing.assert_close(field_spectral, expected, rtol=0, atol=1e-14)


def test_to_physical_keeps_the_nyquist_mode_with_explicit_sizes() -> None:
    mesh = make_mesh(8, 1)
    field_spectral = torch.zeros(5, dtype=torch.complex128)
    field_spectral[4] = 1.0
    field_physical = mesh.decomposition.to_physical(field_spectral)
    expected = torch.tensor([1.0, -1.0] * 4, dtype=torch.float64)
    torch.testing.assert_close(field_physical, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_sum_and_max_reduce_the_spatial_axes_only(dimension: int) -> None:
    mesh = make_mesh(4, dimension)
    field_physical = random_physical(mesh, 2, 3)
    spatial_axes = tuple(range(-dimension, 0))
    total = mesh.decomposition.sum(field_physical)
    largest = mesh.decomposition.max(field_physical)
    assert total.shape == (2, 3)
    assert largest.shape == (2, 3)
    assert total.device == field_physical.device
    assert largest.device == field_physical.device
    torch.testing.assert_close(total, field_physical.sum(spatial_axes))
    torch.testing.assert_close(largest, field_physical.amax(spatial_axes))


def test_serial_rank_and_world_size() -> None:
    decomposition = make_mesh(4, 3).decomposition
    assert decomposition.rank == 0
    assert decomposition.world_size == 1


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_gather_then_scatter_returns_the_local_field(dimension: int) -> None:
    mesh = make_mesh(8, dimension)
    field_spectral = mesh.decomposition.to_spectral(random_physical(mesh, 3))
    gathered = mesh.decomposition.gather(field_spectral)
    assert gathered.shape == (3, *((dimension - 1) * (8,) + (5,)))
    torch.testing.assert_close(mesh.decomposition.scatter(gathered), field_spectral)

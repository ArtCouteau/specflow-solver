import math

import pytest
import torch

from specflow.decomposition import SerialDecomposition
from specflow.mesh import Mesh

DIMENSIONS = [1, 2, 3]
REAL_DTYPES = [torch.float64, torch.float32]
DEVICES = [
    torch.device("cpu"),
    torch.device("meta"),
    pytest.param(torch.device("cuda"), marks=pytest.mark.gpu),
]
COMPLEX_DTYPES = {torch.float64: torch.complex128, torch.float32: torch.complex64}


def make_mesh(
    points: int,
    dimension: int,
    length: float = 2 * math.pi,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> Mesh:
    decomposition = SerialDecomposition(points, dimension)
    return Mesh(points, length, dimension, decomposition, dtype=dtype, device=device)


def mesh_arrays(mesh: Mesh) -> dict[str, torch.Tensor]:
    return {
        "wavenumbers": mesh.wavenumbers,
        "derivative_wavenumbers": mesh.derivative_wavenumbers,
        "wavenumbers_squared": mesh.wavenumbers_squared,
        "inverse_wavenumbers_squared": mesh.inverse_wavenumbers_squared,
        "parseval_weights": mesh.parseval_weights,
        "coordinates": mesh.coordinates,
    }


@pytest.mark.parametrize("points", [0, 7, 15])
def test_odd_number_of_points_raises(points: int) -> None:
    with pytest.raises(ValueError, match="even"):
        make_mesh(points, 2)


@pytest.mark.parametrize("dimension", [0, 4])
def test_dimension_outside_one_to_three_raises(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimension"):
        make_mesh(8, dimension)


@pytest.mark.parametrize("dtype", [torch.complex128, torch.int64, torch.float16])
def test_dtype_other_than_float32_or_float64_raises(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="dtype"):
        make_mesh(8, 2, dtype=dtype)


@pytest.mark.parametrize(
    "decomposition", [SerialDecomposition(16, 2), SerialDecomposition(8, 3)]
)
def test_decomposition_of_another_mesh_raises(
    decomposition: SerialDecomposition,
) -> None:
    with pytest.raises(ValueError, match="decomposition"):
        Mesh(8, 1.0, 2, decomposition)


def test_float64_is_the_default_precision() -> None:
    mesh = Mesh(8, 1.0, 2, SerialDecomposition(8, 2))
    assert mesh.dtype == torch.float64
    assert mesh.complex_dtype == torch.complex128
    assert mesh.device == torch.device("cpu")


@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_every_mesh_array_is_in_the_mesh_dtype_and_device(
    dimension: int, dtype: torch.dtype, device: torch.device
) -> None:
    mesh = make_mesh(8, dimension, length=1.0, dtype=dtype, device=device)
    assert mesh.complex_dtype == COMPLEX_DTYPES[dtype]
    for name, array in mesh_arrays(mesh).items():
        assert array.dtype == dtype, name
        assert array.device.type == device.type, name


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_float64_wavenumbers_are_exact_to_double_precision(dimension: int) -> None:
    mesh = make_mesh(16, dimension, length=0.1)
    integers = mesh.wavenumbers * 0.1 / (2 * math.pi)
    torch.testing.assert_close(integers, integers.round(), rtol=0, atol=1e-13)
    torch.testing.assert_close(
        integers[-1, ..., 3], torch.full_like(integers[-1, ..., 3], 3.0)
    )


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_mesh_shapes_follow_the_layout(dimension: int) -> None:
    mesh = make_mesh(8, dimension)
    spectral = (dimension - 1) * (8,) + (5,)
    physical = dimension * (8,)
    assert mesh.spectral_shape == spectral
    assert mesh.physical_shape == physical
    assert mesh.wavenumbers.shape == (dimension, *spectral)
    assert mesh.wavenumbers_squared.shape == spectral
    assert mesh.inverse_wavenumbers_squared.shape == spectral
    assert mesh.parseval_weights.shape == spectral
    assert mesh.coordinates.shape == (dimension, *physical)


def test_wavenumbers_are_two_pi_over_length_times_the_integer() -> None:
    mesh = make_mesh(8, 2, length=0.5)
    n = torch.tensor([0, 1, 2, 3, -4, -3, -2, -1], dtype=torch.float64)
    n_last = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float64)
    scale = 2 * math.pi / 0.5
    torch.testing.assert_close(mesh.wavenumbers[0, :, 0], scale * n)
    torch.testing.assert_close(mesh.wavenumbers[1, 0, :], scale * n_last)
    torch.testing.assert_close(
        mesh.wavenumbers_squared, mesh.wavenumbers.square().sum(0)
    )


def test_inverse_wavenumbers_squared_inverts_the_derivative_wavenumbers() -> None:
    mesh = make_mesh(8, 3)
    inverse = mesh.inverse_wavenumbers_squared
    derivative_squared = mesh.derivative_wavenumbers.square().sum(dim=0)
    nonzero = derivative_squared > 0
    assert inverse[0, 0, 0].item() == 0.0
    assert torch.all(inverse[~nonzero] == 0)
    torch.testing.assert_close(
        (inverse * derivative_squared)[nonzero], torch.ones_like(inverse)[nonzero]
    )


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_parseval_weights_give_the_mean_square(dimension: int) -> None:
    mesh = make_mesh(8, dimension)
    generator = torch.Generator().manual_seed(1)
    field_physical = torch.randn(mesh.physical_shape, generator=generator).double()
    field_spectral = mesh.decomposition.to_spectral(field_physical)
    parseval = (mesh.parseval_weights * field_spectral.abs().square()).sum()
    torch.testing.assert_close(parseval, field_physical.square().mean())


def test_coordinates_are_the_equispaced_points() -> None:
    mesh = make_mesh(4, 3, length=2.0)
    points = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64)
    torch.testing.assert_close(mesh.coordinates[0, :, 1, 2], points)
    torch.testing.assert_close(mesh.coordinates[1, 3, :, 0], points)
    torch.testing.assert_close(mesh.coordinates[2, 0, 2, :], points)
    assert mesh.spacing == 0.5

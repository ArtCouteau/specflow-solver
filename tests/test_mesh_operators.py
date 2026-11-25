import math

import pytest
import torch

from specflow.decomposition import SerialDecomposition
from specflow.mesh import Mesh

DIMENSIONS = [1, 2, 3]
TOLERANCE = 1e-12


def make_mesh(
    points: int,
    dimension: int,
    length: float = 2 * math.pi,
    dtype: torch.dtype = torch.float64,
) -> Mesh:
    decomposition = SerialDecomposition(points, dimension)
    return Mesh(points, length, dimension, decomposition, dtype=dtype)


def mode(mesh: Mesh, integers: tuple[int, ...]) -> torch.Tensor:
    shape = (mesh.dimension,) + mesh.dimension * (1,)
    wavevector = torch.tensor(integers, dtype=mesh.dtype).reshape(shape)
    return (2 * math.pi / mesh.length) * wavevector


def phase(mesh: Mesh, integers: tuple[int, ...]) -> torch.Tensor:
    return (mode(mesh, integers) * mesh.coordinates).sum(0)


def spectral(mesh: Mesh, field_physical: torch.Tensor) -> torch.Tensor:
    return mesh.decomposition.to_spectral(field_physical)


def physical(mesh: Mesh, field_spectral: torch.Tensor) -> torch.Tensor:
    return mesh.decomposition.to_physical(field_spectral)


def random_vector_spectral(mesh: Mesh, *leading: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (*leading, mesh.dimension, *mesh.physical_shape)
    field_physical = torch.randn(shape, generator=generator, dtype=mesh.dtype)
    return spectral(mesh, field_physical)


def assert_zero(field: torch.Tensor) -> None:
    assert field.abs().max().item() <= TOLERANCE


MODES = {1: (3,), 2: (1, -2), 3: (2, -1, 3)}


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_gradient_of_a_single_mode(dimension: int) -> None:
    mesh = make_mesh(16, dimension, length=0.7)
    integers = MODES[dimension]
    scalar_physical = torch.cos(phase(mesh, integers)).expand(2, *mesh.physical_shape)
    gradient_spectral = mesh.gradient(spectral(mesh, scalar_physical))
    assert gradient_spectral.shape == (2, dimension, *mesh.spectral_shape)
    expected = -mode(mesh, integers) * torch.sin(phase(mesh, integers))
    torch.testing.assert_close(
        physical(mesh, gradient_spectral), expected.expand(2, *expected.shape)
    )


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_gradient_of_a_vector_adds_the_derivative_axis_last(dimension: int) -> None:
    mesh = make_mesh(16, dimension, length=0.7)
    integers = MODES[dimension]
    amplitude = torch.arange(1, dimension + 1, dtype=torch.float64)
    amplitude = amplitude.reshape((dimension,) + dimension * (1,))
    velocity_physical = amplitude * torch.sin(phase(mesh, integers))
    gradient_physical = physical(mesh, mesh.gradient(spectral(mesh, velocity_physical)))
    assert gradient_physical.shape == (dimension, dimension, *mesh.physical_shape)
    cosine = torch.cos(phase(mesh, integers))
    wavevector = mode(mesh, integers)
    for i in range(dimension):
        for j in range(dimension):
            torch.testing.assert_close(
                gradient_physical[i, j], amplitude[i, 0] * wavevector[j, 0] * cosine
            )


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_divergence_of_a_single_mode(dimension: int) -> None:
    mesh = make_mesh(16, dimension, length=3.0)
    integers = MODES[dimension]
    amplitude = torch.linspace(-1, 2, dimension, dtype=torch.float64)
    amplitude = amplitude.reshape((dimension,) + dimension * (1,))
    velocity_physical = amplitude * torch.cos(phase(mesh, integers))
    divergence_spectral = mesh.divergence(spectral(mesh, velocity_physical))
    assert divergence_spectral.shape == mesh.spectral_shape
    expected = -(amplitude * mode(mesh, integers)).sum(0) * torch.sin(
        phase(mesh, integers)
    )
    torch.testing.assert_close(physical(mesh, divergence_spectral), expected)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_laplacian_of_a_single_mode(dimension: int) -> None:
    mesh = make_mesh(16, dimension, length=0.5)
    integers = MODES[dimension]
    scalar_physical = torch.sin(phase(mesh, integers))
    laplacian_spectral = mesh.laplacian(spectral(mesh, scalar_physical))
    expected = -mode(mesh, integers).square().sum() * scalar_physical
    torch.testing.assert_close(physical(mesh, laplacian_spectral), expected)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_laplacian_is_the_divergence_of_the_gradient_without_nyquist(
    dimension: int,
) -> None:
    mesh = make_mesh(8, dimension)
    resolved = (mesh.derivative_wavenumbers == mesh.wavenumbers).all(dim=0)
    velocity_spectral = random_vector_spectral(mesh, 2) * resolved
    torch.testing.assert_close(
        mesh.divergence(mesh.gradient(velocity_spectral)),
        mesh.laplacian(velocity_spectral),
    )


def test_curl_2d_of_the_taylor_green_vortex_is_its_scalar_vorticity() -> None:
    mesh = make_mesh(16, 2)
    x, y = mesh.coordinates
    velocity_physical = torch.stack(
        [torch.sin(x) * torch.cos(y), -torch.cos(x) * torch.sin(y)]
    ).expand(3, 2, 16, 16)
    vorticity_spectral = mesh.curl_2d(spectral(mesh, velocity_physical))
    assert vorticity_spectral.shape == (3, *mesh.spectral_shape)
    expected = 2 * torch.sin(x) * torch.sin(y)
    torch.testing.assert_close(
        physical(mesh, vorticity_spectral), expected.expand(3, 16, 16)
    )


def test_curl_3d_of_a_single_mode_field() -> None:
    mesh = make_mesh(8, 3)
    x, y, z = mesh.coordinates
    velocity_physical = torch.stack([torch.sin(y), torch.sin(z), torch.sin(x)])
    vorticity_spectral = mesh.curl_3d(spectral(mesh, velocity_physical))
    assert vorticity_spectral.shape == (3, *mesh.spectral_shape)
    expected = -torch.stack([torch.cos(z), torch.cos(x), torch.cos(y)])
    torch.testing.assert_close(physical(mesh, vorticity_spectral), expected)


@pytest.mark.parametrize("dimension", [1, 3])
def test_curl_2d_outside_two_dimensions_raises(dimension: int) -> None:
    mesh = make_mesh(4, dimension)
    with pytest.raises(ValueError, match="2D"):
        mesh.curl_2d(random_vector_spectral(mesh))


@pytest.mark.parametrize("dimension", [1, 2])
def test_curl_3d_outside_three_dimensions_raises(dimension: int) -> None:
    mesh = make_mesh(4, dimension)
    with pytest.raises(ValueError, match="3D"):
        mesh.curl_3d(random_vector_spectral(mesh))


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_projection_is_idempotent(dimension: int) -> None:
    mesh = make_mesh(8, dimension)
    projected_spectral = mesh.project(random_vector_spectral(mesh, 2))
    torch.testing.assert_close(mesh.project(projected_spectral), projected_spectral)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_projection_leaves_a_divergence_free_field(dimension: int) -> None:
    mesh = make_mesh(8, dimension)
    projected_spectral = mesh.project(random_vector_spectral(mesh, 2))
    assert_zero(mesh.divergence(projected_spectral))


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_projection_removes_gradients(dimension: int) -> None:
    mesh = make_mesh(8, dimension, length=1.0)
    potential_spectral = random_vector_spectral(mesh)[0]
    gradient_spectral = mesh.gradient(potential_spectral)
    scale = gradient_spectral.abs().max().item()
    assert mesh.project(gradient_spectral).abs().max().item() <= TOLERANCE * scale


def test_projection_preserves_a_divergence_free_field_in_2d() -> None:
    mesh = make_mesh(16, 2)
    x, y = mesh.coordinates
    velocity_physical = torch.stack(
        [torch.sin(x) * torch.cos(3 * y), -torch.cos(x) * torch.sin(3 * y) / 3]
    )
    velocity_spectral = spectral(mesh, velocity_physical)
    torch.testing.assert_close(mesh.project(velocity_spectral), velocity_spectral)


def test_projection_preserves_a_curl_in_3d() -> None:
    mesh = make_mesh(8, 3)
    velocity_spectral = mesh.curl_3d(random_vector_spectral(mesh, 2))
    torch.testing.assert_close(mesh.project(velocity_spectral), velocity_spectral)


def test_projection_keeps_the_mean() -> None:
    mesh = make_mesh(8, 3)
    velocity_spectral = random_vector_spectral(mesh)
    torch.testing.assert_close(
        mesh.project(velocity_spectral)[:, 0, 0, 0], velocity_spectral[:, 0, 0, 0]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_operators_keep_the_complex_dtype_of_the_mesh(dtype: torch.dtype) -> None:
    mesh = make_mesh(8, 3, dtype=dtype)
    velocity_spectral = random_vector_spectral(mesh)
    assert velocity_spectral.dtype == mesh.complex_dtype
    results = {
        "gradient": mesh.gradient(velocity_spectral),
        "divergence": mesh.divergence(velocity_spectral),
        "laplacian": mesh.laplacian(velocity_spectral),
        "curl_3d": mesh.curl_3d(velocity_spectral),
        "project": mesh.project(velocity_spectral),
    }
    for name, result in results.items():
        assert result.dtype == mesh.complex_dtype, name


def test_curl_2d_keeps_the_complex_dtype_of_a_float32_mesh() -> None:
    mesh = make_mesh(8, 2, dtype=torch.float32)
    assert mesh.curl_2d(random_vector_spectral(mesh)).dtype == torch.complex64


def test_operators_do_not_modify_their_argument() -> None:
    mesh = make_mesh(8, 3)
    velocity_spectral = random_vector_spectral(mesh)
    original = velocity_spectral.clone()
    mesh.gradient(velocity_spectral)
    mesh.divergence(velocity_spectral)
    mesh.laplacian(velocity_spectral)
    mesh.curl_3d(velocity_spectral)
    mesh.project(velocity_spectral)
    torch.testing.assert_close(velocity_spectral, original, rtol=0, atol=0)


def nyquist_field_spectral(mesh: Mesh, axis: int) -> torch.Tensor:
    integers = tuple(
        mesh.points // 2 if a == axis else 0 for a in range(mesh.dimension)
    )
    return spectral(mesh, torch.cos(phase(mesh, integers)))


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_derivative_wavenumbers_drop_only_the_nyquist_mode(dimension: int) -> None:
    mesh = make_mesh(8, dimension)
    nyquist = mesh.wavenumbers.abs() == math.pi * mesh.points / mesh.length
    assert torch.all(mesh.derivative_wavenumbers[nyquist] == 0)
    assert torch.equal(
        mesh.derivative_wavenumbers[~nyquist], mesh.wavenumbers[~nyquist]
    )


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_gradient_of_a_nyquist_mode_is_zero(dimension: int) -> None:
    mesh = make_mesh(8, dimension)
    for axis in range(dimension):
        gradient_spectral = mesh.gradient(nyquist_field_spectral(mesh, axis))
        assert gradient_spectral.abs().max() < TOLERANCE


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_divergence_of_a_nyquist_mode_is_zero(dimension: int) -> None:
    mesh = make_mesh(8, dimension)
    for axis in range(dimension):
        velocity_spectral = torch.zeros(
            (dimension, *mesh.spectral_shape), dtype=mesh.complex_dtype
        )
        velocity_spectral[axis] = nyquist_field_spectral(mesh, axis)
        assert mesh.divergence(velocity_spectral).abs().max() < TOLERANCE


def test_curl_of_a_nyquist_mode_is_zero() -> None:
    mesh = make_mesh(8, 3)
    velocity_spectral = torch.zeros((3, *mesh.spectral_shape), dtype=mesh.complex_dtype)
    velocity_spectral[1] = nyquist_field_spectral(mesh, 0)
    assert mesh.curl_3d(velocity_spectral).abs().max() < TOLERANCE

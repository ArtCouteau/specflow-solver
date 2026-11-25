import math

import torch

from specflow.dealiasing import TwoThirdsDealiasing
from specflow.decomposition import SerialDecomposition
from specflow.mesh import Mesh

TOLERANCE = 1e-14


def make_mesh(points: int, dimension: int) -> Mesh:
    decomposition = SerialDecomposition(points, dimension)
    return Mesh(points, 2 * math.pi, dimension, decomposition)


def cosine(mesh: Mesh, integers: tuple[int, ...]) -> torch.Tensor:
    wavevector = torch.tensor(integers, dtype=mesh.dtype)
    wavevector = wavevector.reshape(mesh.dimension, *mesh.dimension * (1,))
    cosine_physical = torch.cos((wavevector * mesh.coordinates).sum(0))
    return mesh.decomposition.to_spectral(cosine_physical)


def product(
    mesh: Mesh, a_spectral: torch.Tensor, b_spectral: torch.Tensor
) -> torch.Tensor:
    dealiasing = TwoThirdsDealiasing(mesh)
    return dealiasing.product(torch.mul, a_spectral, b_spectral)


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert (actual - expected).abs().max().item() <= TOLERANCE


def test_a_band_limited_product_below_the_cutoff_is_exact() -> None:
    mesh = make_mesh(12, 1)

    result_spectral = product(mesh, cosine(mesh, (1,)), cosine(mesh, (2,)))

    expected_spectral = (cosine(mesh, (1,)) + cosine(mesh, (3,))) / 2
    assert_close(result_spectral, expected_spectral)


def test_the_cutoff_is_a_strict_inequality() -> None:
    mesh = make_mesh(12, 1)

    result_spectral = product(mesh, cosine(mesh, (2,)), cosine(mesh, (2,)))

    assert_close(result_spectral, cosine(mesh, (0,)) / 2)


def test_an_aliased_product_is_removed_exactly() -> None:
    mesh = make_mesh(10, 1)

    result_spectral = product(mesh, cosine(mesh, (3,)), cosine(mesh, (3,)))

    assert_close(result_spectral, cosine(mesh, (0,)) / 2)


def test_the_truncation_is_spherical() -> None:
    mesh = make_mesh(12, 2)

    result_spectral = product(mesh, cosine(mesh, (3, 0)), cosine(mesh, (0, 3)))
    kept_spectral = product(mesh, cosine(mesh, (3, 0)), cosine(mesh, (0, 1)))

    assert_close(result_spectral, torch.zeros(()))
    expected_spectral = (cosine(mesh, (3, 1)) + cosine(mesh, (3, -1))) / 2
    assert_close(kept_spectral, expected_spectral)


def test_a_product_of_one_factor_transforms_it_once() -> None:
    mesh = make_mesh(12, 3)
    dealiasing = TwoThirdsDealiasing(mesh)
    a_spectral = cosine(mesh, (1, 2, 0))

    square_spectral = dealiasing.product(torch.square, a_spectral)

    assert_close(square_spectral, product(mesh, a_spectral, a_spectral))

import subprocess
import sys

import pytest
import torch

from specflow import algebra

DIMENSIONS = [1, 2, 3]


def random_field(*shape: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float64)


def unit_vectors(dimension: int, points: int) -> list[torch.Tensor]:
    spatial = dimension * (points,)
    vectors: list[torch.Tensor] = []
    for component in range(dimension):
        vector = torch.zeros(dimension, *spatial, dtype=torch.float64)
        vector[component] = 1.0
        vectors.append(vector)
    return vectors


def test_algebra_does_not_import_the_mesh() -> None:
    code = "import sys, specflow.algebra; print('specflow.mesh' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_inner_contracts_the_component_axis_only(dimension: int) -> None:
    spatial = dimension * (4,)
    a = random_field(5, dimension, *spatial, seed=0)
    b = random_field(5, dimension, *spatial, seed=1)
    result = algebra.inner(a, b, dimension)
    assert result.shape == (5, *spatial)
    expected = sum(a[:, i] * b[:, i] for i in range(dimension))
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_outer_puts_the_first_factor_on_the_rows(dimension: int) -> None:
    spatial = dimension * (4,)
    a = random_field(2, dimension, *spatial, seed=0)
    b = random_field(2, dimension, *spatial, seed=1)
    c = random_field(2, dimension, *spatial, seed=2)
    product = algebra.outer(a, b, dimension)
    assert product.shape == (2, dimension, dimension, *spatial)
    torch.testing.assert_close(product[:, -1, 0], a[:, -1] * b[:, 0])
    torch.testing.assert_close(
        algebra.inner(product, c.unsqueeze(-dimension - 2), dimension),
        a * algebra.inner(b, c, dimension).unsqueeze(-dimension - 1),
    )


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_transpose_swaps_the_component_axes(dimension: int) -> None:
    spatial = dimension * (4,)
    a = random_field(3, dimension, *spatial, seed=0)
    b = random_field(3, dimension, *spatial, seed=1)
    torch.testing.assert_close(
        algebra.transpose(algebra.outer(a, b, dimension), dimension),
        algebra.outer(b, a, dimension),
    )


def test_cross_2d_of_unit_vectors_is_one() -> None:
    e_x, e_y = unit_vectors(2, 4)
    torch.testing.assert_close(algebra.cross_2d(e_x, e_y), torch.ones(4, 4).double())
    torch.testing.assert_close(algebra.cross_2d(e_y, e_x), -torch.ones(4, 4).double())


def test_cross_2d_is_antisymmetric_with_leading_axes() -> None:
    a = random_field(3, 2, 4, 4, seed=0)
    b = random_field(3, 2, 4, 4, seed=1)
    result = algebra.cross_2d(a, b)
    assert result.shape == (3, 4, 4)
    torch.testing.assert_close(result, a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    torch.testing.assert_close(algebra.cross_2d(b, a), -result)


def test_cross_3d_of_unit_vectors_is_right_handed() -> None:
    e_x, e_y, e_z = unit_vectors(3, 4)
    torch.testing.assert_close(algebra.cross_3d(e_x, e_y), e_z)
    torch.testing.assert_close(algebra.cross_3d(e_y, e_z), e_x)
    torch.testing.assert_close(algebra.cross_3d(e_z, e_x), e_y)


def test_cross_3d_is_orthogonal_to_its_factors_with_leading_axes() -> None:
    a = random_field(2, 3, 4, 4, 4, seed=0)
    b = random_field(2, 3, 4, 4, 4, seed=1)
    result = algebra.cross_3d(a, b)
    assert result.shape == (2, 3, 4, 4, 4)
    zero = torch.zeros(2, 4, 4, 4, dtype=torch.float64)
    torch.testing.assert_close(algebra.inner(result, a, 3), zero)
    torch.testing.assert_close(algebra.inner(result, b, 3), zero)
    torch.testing.assert_close(algebra.cross_3d(b, a), -result)

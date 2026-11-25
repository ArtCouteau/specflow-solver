import math

import pytest

from specflow.decomposition import SerialDecomposition
from specflow.mesh import Mesh


def make_mesh(points: int, dimension: int, length: float = 2 * math.pi) -> Mesh:
    return Mesh(points, length, dimension, SerialDecomposition(points, dimension))


@pytest.mark.parametrize("points", [7, 15])
def test_odd_number_of_points_raises(points: int) -> None:
    with pytest.raises(ValueError, match="even"):
        make_mesh(points, 2)

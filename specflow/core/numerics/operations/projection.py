import torch

from . import algebra, derivative, rank
from .. import context


def to_divergence_free_plane(vector: torch.Tensor) -> torch.Tensor:
    pressureCorrection = algebra.outer_vector_vector(
        context.get().mesh.grid,
        derivative.inv_nabla2()*context.get().mesh.grid
    )

    return algebra.matmul_tensor_vector(
        rank.kronecker_delta_like(pressureCorrection) + pressureCorrection,
        vector
    )

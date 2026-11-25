"""The exact removal of aliasing errors from quadratic products."""

import math
from collections.abc import Callable

import torch

from specflow.mesh import Mesh


class TwoThirdsDealiasing:
    """The 2/3 rule: a spherical truncation of the product to $|n| < N/3$.

    Only the product is truncated, never its factors, so a state band-limited to
    $|n| < N/3$ keeps its products free of aliasing.
    """

    def __init__(self, mesh: Mesh) -> None:
        self._decomposition = mesh.decomposition
        integers = mesh.wavenumbers.mul(mesh.length / (2 * math.pi)).round_()
        integers_squared = integers.square_().sum(dim=0)
        self._mask = (9 * integers_squared < mesh.points**2).to(mesh.dtype)

    def product(
        self, operation: Callable[..., torch.Tensor], *factors_spectral: torch.Tensor
    ) -> torch.Tensor:
        """Return the dealiased product of spectral fields, formed at the mesh points.

        ``operation`` takes the factors in physical space and returns their product.
        """
        factors_physical = [
            self._decomposition.to_physical(factor) for factor in factors_spectral
        ]
        product_physical = operation(*factors_physical)
        return self._decomposition.to_spectral(product_physical).mul_(self._mask)

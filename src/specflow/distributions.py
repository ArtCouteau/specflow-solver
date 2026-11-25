"""Moments and histograms of fields in physical space, and what is formed from them.

Moments and counts are sums over the spatial axes, through the decomposition, and keep
every other axis of the field. They are what a statistic averages; standardized
moments and densities are formed afterwards, from the averages.
"""

import torch

from specflow.mesh import Mesh


def raw_moments(
    mesh: Mesh, field_physical: torch.Tensor, order: int = 4
) -> torch.Tensor:
    r"""Return the raw moments $\langle f^p \rangle$ for $p = 1, \dots, P$.

    $P$ is ``order``. The order axis leads, $p = 1$ first, followed by the
    non-spatial axes of the field.
    """
    moments = field_physical.new_empty(order, *field_physical.shape[: -mesh.dimension])
    power_physical = field_physical.clone()
    for index in range(order):
        if index:
            power_physical.mul_(field_physical)
        moments[index] = mesh.decomposition.sum(power_physical)
    return moments.div_(mesh.points**mesh.dimension)


def histogram(
    mesh: Mesh, field_physical: torch.Tensor, edges: torch.Tensor
) -> torch.Tensor:
    """Return the integer counts of a field over the bins between increasing edges.

    The bins $[e_i, e_{i+1})$ are half-open and form a last axis, after an underflow
    count of the values below the first edge and before an overflow count of the
    values at or above the last edge.
    """
    bins = torch.bucketize(field_physical, edges, right=True)
    ones = torch.ones_like(field_physical, dtype=torch.int64)
    return mesh.decomposition.binned_sum(ones, bins, edges.numel() + 1)


def standardized_moments(
    moments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Return the mean, variance, skewness and flatness from raw moments.

    The skewness is $\langle f'^3 \rangle / \langle f'^2 \rangle^{3/2}$ and the
    flatness $\langle f'^4 \rangle / \langle f'^2 \rangle^2$. Only the first four
    orders are read.
    """
    if moments.shape[0] < 4:
        raise ValueError(
            f"standardized moments need the raw moments of orders 1 to 4, "
            f"got {moments.shape[0]}"
        )
    first, second, third, fourth = moments[:4]
    mean_square = first.square()
    variance = second - mean_square
    third_central = third - 3 * first * second + 2 * first * mean_square
    fourth_central = (
        fourth - 4 * first * third + 6 * mean_square * second - 3 * mean_square.square()
    )
    skewness = third_central.div_(variance.pow(1.5))
    flatness = fourth_central.div_(variance.square())
    return first.clone(), variance, skewness, flatness


def probability_density(counts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Return the probability density in each bin from counts over edges.

    The counts carry the underflow and overflow of ``histogram``, which enter the
    total but have no density, so that the density integrates to the fraction of
    values within the edges.
    """
    total = counts.sum(-1, keepdim=True).to(edges.dtype)
    density = counts[..., 1:-1].to(edges.dtype, copy=True)
    return density.div_(total).div_(edges.diff())

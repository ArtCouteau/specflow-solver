"""Pointwise products of vector and tensor fields.

Fields follow the layout: the ``dimension`` spatial axes last, component axes just
before them, leading axes free. The functions work in either space.
"""

import torch


def inner(a: torch.Tensor, b: torch.Tensor, dimension: int) -> torch.Tensor:
    """Return the contraction of the last component axes of two fields."""
    return (a * b).sum(dim=-dimension - 1)


def outer(a: torch.Tensor, b: torch.Tensor, dimension: int) -> torch.Tensor:
    """Return the tensor field a_i b_j, with i on the first component axis."""
    return a.unsqueeze(-dimension - 1) * b.unsqueeze(-dimension - 2)


def transpose(tensor: torch.Tensor, dimension: int) -> torch.Tensor:
    """Return a view of a tensor field with its two component axes swapped."""
    return tensor.transpose(-dimension - 2, -dimension - 1)


def cross_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the scalar a_x b_y - a_y b_x of two 2D vector fields."""
    return a[..., 0, :, :] * b[..., 1, :, :] - a[..., 1, :, :] * b[..., 0, :, :]


def cross_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the vector a x b of two 3D vector fields."""
    a_x, a_y, a_z = a.unbind(dim=-4)
    b_x, b_y, b_z = b.unbind(dim=-4)
    return torch.stack(
        [a_y * b_z - a_z * b_y, a_z * b_x - a_x * b_z, a_x * b_y - a_y * b_x], dim=-4
    )

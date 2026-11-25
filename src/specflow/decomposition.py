"""The split of the mesh across ranks, through which FFTs and spatial reductions go.

Spatial axes are the last ``dimension`` axes of a field; the leading axes are free.
"""

from dataclasses import dataclass
from typing import Protocol

import torch


class Decomposition(Protocol):
    """The interface every decomposition provides to the mesh and the physics."""

    @property
    def points(self) -> int:
        """The global number of points per direction."""
        ...

    @property
    def dimension(self) -> int:
        """The number of spatial dimensions."""
        ...

    @property
    def rank(self) -> int:
        """The rank of this process."""
        ...

    @property
    def world_size(self) -> int:
        """The number of ranks."""
        ...

    @property
    def spectral_slices(self) -> tuple[slice, ...]:
        """The local index ranges of the global spectral axes, one per spatial axis."""
        ...

    @property
    def physical_slices(self) -> tuple[slice, ...]:
        """The local index ranges of the global physical axes, one per spatial axis."""
        ...

    def to_spectral(self, field_physical: torch.Tensor) -> torch.Tensor:
        """Return the Fourier coefficients of a field, with forward normalization."""
        ...

    def to_physical(self, field_spectral: torch.Tensor) -> torch.Tensor:
        """Return a field at the mesh points, with forward normalization."""
        ...

    def sum(self, field: torch.Tensor) -> torch.Tensor:
        """Return the sum over the spatial axes, on the field's device."""
        ...

    def max(self, field: torch.Tensor) -> torch.Tensor:
        """Return the maximum over the spatial axes, on the field's device."""
        ...

    def gather(self, field_spectral: torch.Tensor) -> torch.Tensor:
        """Return the full field on rank 0, from the local part on each rank."""
        ...

    def scatter(self, field_spectral: torch.Tensor) -> torch.Tensor:
        """Return each rank's local part of a full field held on rank 0."""
        ...


@dataclass(frozen=True)
class SerialDecomposition:
    """The decomposition that gives the whole mesh to rank 0."""

    points: int
    dimension: int

    @property
    def rank(self) -> int:
        """The rank of this process, always 0."""
        return 0

    @property
    def world_size(self) -> int:
        """The number of ranks, always 1."""
        return 1

    @property
    def spectral_slices(self) -> tuple[slice, ...]:
        """The whole spectral axes: N wavenumbers, and N/2 + 1 along the last axis."""
        return (self.dimension - 1) * (slice(0, self.points),) + (
            slice(0, self.points // 2 + 1),
        )

    @property
    def physical_slices(self) -> tuple[slice, ...]:
        """The whole physical axes: N points each."""
        return self.dimension * (slice(0, self.points),)

    def to_spectral(self, field_physical: torch.Tensor) -> torch.Tensor:
        """Return the Fourier coefficients of a field, with forward normalization.

        Expanded and strided views are copied to contiguous memory first.
        """
        return torch.fft.rfftn(
            field_physical.contiguous(), s=self._sizes, dim=self._axes, norm="forward"
        )

    def to_physical(self, field_spectral: torch.Tensor) -> torch.Tensor:
        """Return a field at the mesh points, with forward normalization.

        Expanded and strided views are copied to contiguous memory first.
        """
        return torch.fft.irfftn(
            field_spectral.contiguous(), s=self._sizes, dim=self._axes, norm="forward"
        )

    def sum(self, field: torch.Tensor) -> torch.Tensor:
        """Return the sum over the spatial axes, on the field's device."""
        return field.sum(dim=self._axes)

    def max(self, field: torch.Tensor) -> torch.Tensor:
        """Return the maximum over the spatial axes, on the field's device."""
        return field.amax(dim=self._axes)

    def gather(self, field_spectral: torch.Tensor) -> torch.Tensor:
        """Return the field itself, since rank 0 holds all of it."""
        return field_spectral

    def scatter(self, field_spectral: torch.Tensor) -> torch.Tensor:
        """Return the field itself, since rank 0 holds all of it."""
        return field_spectral

    @property
    def _axes(self) -> tuple[int, ...]:
        return tuple(range(-self.dimension, 0))

    @property
    def _sizes(self) -> tuple[int, ...]:
        return self.dimension * (self.points,)

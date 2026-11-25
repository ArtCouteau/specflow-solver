"""The mesh of the periodic domain and its spectral derivatives."""

import math
from dataclasses import dataclass, field

import torch

from specflow import algebra
from specflow.decomposition import Decomposition

COMPLEX_DTYPES = {torch.float32: torch.complex64, torch.float64: torch.complex128}


@dataclass(frozen=True, eq=False)
class Mesh:
    """N equispaced points per direction of the periodic domain [0, L)^d.

    The arrays are local to this rank, in the mesh's dtype and on its device:
    ``wavenumbers`` and ``derivative_wavenumbers`` (d, *spectral_shape), the latter
    with the Nyquist mode zeroed for first derivatives, ``wavenumbers_squared``,
    ``inverse_wavenumbers_squared`` (of the derivative wavenumbers, 0 where they
    vanish) and ``parseval_weights``
    (*spectral_shape), and ``coordinates`` (d, *physical_shape).
    """

    points: int
    length: float
    dimension: int
    decomposition: Decomposition
    dtype: torch.dtype = torch.float64
    device: torch.device = torch.device("cpu")
    wavenumbers: torch.Tensor = field(init=False, repr=False)
    derivative_wavenumbers: torch.Tensor = field(init=False, repr=False)
    wavenumbers_squared: torch.Tensor = field(init=False, repr=False)
    inverse_wavenumbers_squared: torch.Tensor = field(init=False, repr=False)
    parseval_weights: torch.Tensor = field(init=False, repr=False)
    coordinates: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate()
        integers = self._local_integers()
        wavenumbers = integers.to(self.dtype).mul_(2 * math.pi / self.length)
        derivative_wavenumbers = wavenumbers.masked_fill(
            integers.abs() == self.points // 2, 0.0
        )
        wavenumbers_squared = wavenumbers.square().sum(dim=0)
        derivative_wavenumbers_squared = derivative_wavenumbers.square().sum(dim=0)
        inverse_wavenumbers_squared = derivative_wavenumbers_squared.reciprocal()
        inverse_wavenumbers_squared.masked_fill_(
            derivative_wavenumbers_squared == 0, 0.0
        )
        object.__setattr__(self, "wavenumbers", wavenumbers)
        object.__setattr__(self, "derivative_wavenumbers", derivative_wavenumbers)
        object.__setattr__(self, "wavenumbers_squared", wavenumbers_squared)
        object.__setattr__(
            self, "inverse_wavenumbers_squared", inverse_wavenumbers_squared
        )
        object.__setattr__(self, "parseval_weights", self._local_parseval_weights())
        object.__setattr__(self, "coordinates", self._local_coordinates())

    @property
    def complex_dtype(self) -> torch.dtype:
        """The dtype of spectral fields, the complex counterpart of the mesh dtype."""
        return COMPLEX_DTYPES[self.dtype]

    @property
    def spacing(self) -> float:
        """The distance between neighbouring mesh points, L/N."""
        return self.length / self.points

    @property
    def spectral_shape(self) -> tuple[int, ...]:
        """The local shape of the spatial axes of a spectral field."""
        return tuple(s.stop - s.start for s in self.decomposition.spectral_slices)

    @property
    def physical_shape(self) -> tuple[int, ...]:
        """The local shape of the spatial axes of a physical field."""
        return tuple(s.stop - s.start for s in self.decomposition.physical_slices)

    def gradient(self, field_spectral: torch.Tensor) -> torch.Tensor:
        """Return the gradient, the derivative axis becoming the last component axis.

        A vector field u gives the tensor field ∂ⱼuᵢ, with j on the last component axis.
        """
        derivative_axis = -self.dimension - 1
        gradient_spectral = (
            field_spectral.unsqueeze(derivative_axis) * self.derivative_wavenumbers
        )
        return gradient_spectral.mul_(1j)

    def divergence(self, field_spectral: torch.Tensor) -> torch.Tensor:
        """Return the divergence, contracting the last component axis."""
        return algebra.inner(
            field_spectral, self.derivative_wavenumbers, self.dimension
        ).mul_(1j)

    def laplacian(self, field_spectral: torch.Tensor) -> torch.Tensor:
        """Return the Laplacian, componentwise."""
        return (field_spectral * self.wavenumbers_squared).neg_()

    def curl_2d(self, velocity_spectral: torch.Tensor) -> torch.Tensor:
        """Return the scalar vorticity ∂ₓu_y - ∂_yuₓ of a 2D vector field."""
        if self.dimension != 2:
            raise ValueError(f"curl_2d needs a 2D mesh, got {self.dimension}D")
        return algebra.cross_2d(self.derivative_wavenumbers, velocity_spectral).mul_(1j)

    def curl_3d(self, velocity_spectral: torch.Tensor) -> torch.Tensor:
        """Return the vorticity vector ∇ × u of a 3D vector field."""
        if self.dimension != 3:
            raise ValueError(f"curl_3d needs a 3D mesh, got {self.dimension}D")
        return algebra.cross_3d(self.derivative_wavenumbers, velocity_spectral).mul_(1j)

    def project(self, velocity_spectral: torch.Tensor) -> torch.Tensor:
        """Return the divergence-free part u - k(k·u)/k² of a vector field.

        Built on the derivative wavenumbers, so that its result has zero divergence.
        """
        coefficient_spectral = algebra.inner(
            self.derivative_wavenumbers, velocity_spectral, self.dimension
        ).mul_(self.inverse_wavenumbers_squared)
        longitudinal_spectral = (
            coefficient_spectral.unsqueeze(-self.dimension - 1)
            * self.derivative_wavenumbers
        )
        return longitudinal_spectral.neg_().add_(velocity_spectral)

    def _validate(self) -> None:
        if self.points < 2 or self.points % 2:
            raise ValueError(f"points must be even and at least 2, got {self.points}")
        if self.dimension not in (1, 2, 3):
            raise ValueError(f"dimension must be 1, 2 or 3, got {self.dimension}")
        if self.dtype not in COMPLEX_DTYPES:
            raise ValueError(f"dtype must be float32 or float64, got {self.dtype}")
        decomposition = self.decomposition
        if (decomposition.points, decomposition.dimension) != (
            self.points,
            self.dimension,
        ):
            raise ValueError(
                f"decomposition of a {decomposition.dimension}D mesh with "
                f"{decomposition.points} points, expected a {self.dimension}D mesh "
                f"with {self.points} points"
            )

    def _integers(self) -> torch.Tensor:
        indices = torch.arange(self.points, device=self.device)
        return (indices + self.points // 2) % self.points - self.points // 2

    def _local_integers(self) -> torch.Tensor:
        integers = self._integers()
        last = torch.arange(self.points // 2 + 1, device=self.device)
        slices = self.decomposition.spectral_slices
        axes = [integers[s] for s in slices[:-1]] + [last[slices[-1]]]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"))

    def _local_parseval_weights(self) -> torch.Tensor:
        last = torch.arange(self.points // 2 + 1, device=self.device)
        last = last[self.decomposition.spectral_slices[-1]]
        on_plane = (last == 0) | (last == self.points // 2)
        weights = torch.full(
            self.spectral_shape, 2.0, dtype=self.dtype, device=self.device
        )
        return weights.sub_(on_plane.to(self.dtype))

    def _local_coordinates(self) -> torch.Tensor:
        indices = torch.arange(self.points, device=self.device).to(self.dtype)
        axes = [indices[s] for s in self.decomposition.physical_slices]
        grids = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(grids).mul_(self.spacing)

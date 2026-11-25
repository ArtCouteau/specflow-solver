"""The two kinds of terms of an equation, and the first-pass terms.

A linear term is diagonal in spectral space: its contribution is its multiplier
times the equation's variable, and the multiplier enters the linear operator.
A nonlinear term computes its contribution from the state and the time. Both declare
``reads``, the state keys they read with the component shape expected for each,
``()`` for a scalar and ``(d,)`` for a vector.
"""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch

from specflow import algebra
from specflow.dealiasing import TwoThirdsDealiasing
from specflow.mesh import Mesh

type State = Mapping[str, torch.Tensor]


class LinearTerm(Protocol):
    """A term fully described by one constant multiplier per wavenumber."""

    @property
    def name(self) -> str:
        """The key of the term's contribution."""
        ...

    @property
    def reads(self) -> Mapping[str, tuple[int, ...]]:
        """The component shape of each state key the term reads."""
        ...

    @property
    def multiplier(self) -> torch.Tensor:
        """The multiplier, (*spectral_shape), broadcast over the components."""
        ...


@runtime_checkable
class NonlinearTerm(Protocol):
    """A term evaluated from the state and the time."""

    @property
    def name(self) -> str:
        """The key of the term's contribution."""
        ...

    @property
    def reads(self) -> Mapping[str, tuple[int, ...]]:
        """The component shape of each state key the term reads."""
        ...

    def contribution(self, state: State, time: float) -> torch.Tensor:
        """Return the term's contribution in spectral space, before any projection."""
        ...


type Term = LinearTerm | NonlinearTerm


class Diffusion:
    """The linear term κ∇²q of a variable q, with a constant diffusivity κ."""

    def __init__(
        self,
        mesh: Mesh,
        variable: str,
        diffusivity: float,
        components: tuple[int, ...] = (),
    ) -> None:
        self._variable = variable
        self._components = components
        self._multiplier = mesh.wavenumbers_squared * -diffusivity

    @property
    def name(self) -> str:
        """The key of the term's contribution, ``"diffusion"``."""
        return "diffusion"

    @property
    def reads(self) -> Mapping[str, tuple[int, ...]]:
        """The variable and its component shape."""
        return {self._variable: self._components}

    @property
    def multiplier(self) -> torch.Tensor:
        """The multiplier −κk², (*spectral_shape)."""
        return self._multiplier


class Advection:
    """The nonlinear term of transport by the velocity, in divergence form.

    The velocity u advects itself, −∇·(u⊗u), and any other variable is a scalar θ,
    −∇·(uθ). The product is dealiased; the term is not projected.
    """

    def __init__(
        self,
        mesh: Mesh,
        dealiasing: TwoThirdsDealiasing,
        variable: str,
        velocity: str = "velocity",
    ) -> None:
        self._mesh = mesh
        self._dealiasing = dealiasing
        self._variable = variable
        self._velocity = velocity

    @property
    def name(self) -> str:
        """The key of the term's contribution, ``"advection"``."""
        return "advection"

    @property
    def reads(self) -> Mapping[str, tuple[int, ...]]:
        """The velocity, (d,), and the advected variable, () unless the velocity."""
        return {self._velocity: (self._mesh.dimension,), self._variable: self._shape}

    @property
    def _shape(self) -> tuple[int, ...]:
        if self._variable == self._velocity:
            return (self._mesh.dimension,)
        return ()

    def contribution(self, state: State, time: float) -> torch.Tensor:
        """Return −∇·(u⊗u) or −∇·(uθ) in spectral space."""
        velocity_spectral = state[self._velocity]
        if self._variable == self._velocity:
            flux_spectral = self._dealiasing.product(
                self._velocity_flux, velocity_spectral
            )
        else:
            flux_spectral = self._dealiasing.product(
                self._scalar_flux, state[self._variable], velocity_spectral
            )
        return self._mesh.divergence(flux_spectral).neg_()

    def _velocity_flux(self, velocity_physical: torch.Tensor) -> torch.Tensor:
        return algebra.outer(velocity_physical, velocity_physical, self._mesh.dimension)

    def _scalar_flux(
        self, scalar_physical: torch.Tensor, velocity_physical: torch.Tensor
    ) -> torch.Tensor:
        component_axis = -self._mesh.dimension - 1
        return scalar_physical.unsqueeze(component_axis) * velocity_physical

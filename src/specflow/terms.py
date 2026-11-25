"""The two kinds of terms of an equation, and the first-pass terms.

A linear term is diagonal in spectral space: its contribution is its multiplier
times the equation's variable, and the multiplier enters the linear operator.
A nonlinear term computes its contribution from the state and the time. Both declare
``reads``, the state keys they read with the component shape expected for each,
``()`` for a scalar and ``(d,)`` for a vector.
"""

from collections.abc import Mapping
from typing import Protocol, override, runtime_checkable

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
        """The multiplier of each wavenumber, broadcast over the components."""
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


class Diffusion(LinearTerm):
    r"""The linear term $\alpha \nabla^2 q$ of a variable, with a constant $\alpha$."""

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
    @override
    def name(self) -> str:
        return "diffusion"

    @property
    @override
    def reads(self) -> Mapping[str, tuple[int, ...]]:
        return {self._variable: self._components}

    @property
    @override
    def multiplier(self) -> torch.Tensor:
        return self._multiplier


class Advection(NonlinearTerm):
    r"""The nonlinear term of transport by the velocity, in divergence form.

    The velocity advects itself, $-\partial_j (u_i u_j)$, and any other variable is a
    scalar, $-\partial_j (u_j \theta)$. The product is dealiased; the term is not
    projected.
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
    @override
    def name(self) -> str:
        return "advection"

    @property
    @override
    def reads(self) -> Mapping[str, tuple[int, ...]]:
        return {self._velocity: (self._mesh.dimension,), self._variable: self._shape}

    @property
    def _shape(self) -> tuple[int, ...]:
        if self._variable == self._velocity:
            return (self._mesh.dimension,)
        return ()

    @override
    def contribution(self, state: State, time: float) -> torch.Tensor:
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

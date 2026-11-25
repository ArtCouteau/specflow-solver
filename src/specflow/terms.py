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

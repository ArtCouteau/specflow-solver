"""Quantities of the post-processing graph.

A node is a (name, kind) pair. The name is the physical quantity, the kind its representation: the generic alias of
its wrapper, e.g. Field[Vector, Spectral] or IntegralValue[Scalar].

Conventions:
1. Names are fully expanded PascalCase physics nouns. A name without suffix is a field quantity, in any of the kinds
   related to it by operators.
2. Constants carry the `Constant` suffix (ViscosityConstant, TimeConstant), apart from same-concept fields such as the
   Viscosity field of generalized Newtonian fluids.
3. Per-term equation contributions are `{Equation}{Term}Contribution`: the term's contribution to the time derivative
   of the equation variable, e.g. VelocityAdvectionContribution.
4. Same node, same value: two paths to one node agree pointwise, otherwise the quantities need different names (the
   pseudo-dissipation nu*|grad u|^2 is PseudoDissipation, not Dissipation).
5. Same name, different kinds: related by operators. Spectral-only quantities get their own name, e.g.
   ModalKineticEnergy = |u_k|^2/2, which is not the Fourier transform of KineticEnergy.
6. If two operators map the same kinds to different values, the kinds need an extra type parameter, as Spectrum has
   its measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from specflow.core.physics.wrappers.wrapper import Wrapper
from specflow.processing.numerics.kind import kind_tag

T = TypeVar("T", bound=Wrapper)


@dataclass(frozen=True)
class Node(Generic[T]):
    name: str
    kind: type[T]

    def __str__(self) -> str:
        return f"{self.name}: {kind_tag(self.kind)}"

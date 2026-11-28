"""The evolution law of one variable: a sum of terms, optionally projected."""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence

import torch

from specflow.diagnostics import relative_error
from specflow.mesh import Mesh
from specflow.terms import LinearTerm, NonlinearTerm, State, Term

type Projection = Callable[[torch.Tensor], torch.Tensor]


class Equation:
    """The tendency of one variable: its linear operator plus its nonlinear tendency.

    ``linear_operator`` is the sum of the linear terms' multipliers, zero without
    linear terms. Only the nonlinear tendency and the contributions are projected;
    the linear operator commutes with the projection. ``reads`` merges the terms'
    ``reads``.
    """

    def __init__(
        self,
        variable: str,
        terms: Sequence[Term],
        projection: Projection | None = None,
    ) -> None:
        self.variable = variable
        self.terms = tuple(terms)
        self.projection = projection
        self._linear_terms = [t for t in self.terms if not isinstance(t, NonlinearTerm)]
        self._nonlinear_terms = [t for t in self.terms if isinstance(t, NonlinearTerm)]
        self._validate()
        self.reads = merge_reads(term.reads for term in self.terms)
        self.linear_operator = _sum_multipliers(self._linear_terms)

    def _validate(self) -> None:
        names: set[str] = set()
        for term in self.terms:
            if term.name in names:
                raise ValueError(f"term name {term.name!r} used twice in an equation")
            names.add(term.name)
        for term in self._linear_terms:
            if set(term.reads) != {self.variable}:
                raise ValueError(
                    f"linear term {term.name!r} reads {sorted(term.reads)}, "
                    f"expected only the variable {self.variable!r}"
                )

    def check_projected(self, mesh: Mesh, state: State) -> None:
        """Raise ``ValueError`` if the projection changes the variable in ``state``.

        The tolerance on the relative L2 change is the square root of the machine
        epsilon of the mesh's dtype; a zero variable passes. Without a projection
        there is nothing to check.
        """
        if self.projection is None:
            return
        field_spectral = state[self.variable]
        component_axes = field_spectral.ndim - mesh.dimension
        residual = relative_error(
            mesh, self.projection(field_spectral), field_spectral, component_axes
        ).item()
        if residual > math.sqrt(torch.finfo(mesh.dtype).eps):
            raise ValueError(
                f"initial {self.variable!r} is not divergence-free: the projection "
                f"changes it by {residual:.3e} relative to its L2 norm"
            )

    def nonlinear_tendency(
        self, state: State, time: float, out_spectral: torch.Tensor
    ) -> None:
        """Write the projected sum of nonlinear contributions to ``out_spectral``."""
        out_spectral.zero_()
        if not self._nonlinear_terms:
            return
        for term in self._nonlinear_terms:
            out_spectral.add_(term.contribution(state, time))
        if self.projection is not None:
            out_spectral.copy_(self.projection(out_spectral))

    def contributions(self, state: State, time: float) -> dict[str, torch.Tensor]:
        """Return each term's projected contribution in spectral space, by term name."""
        contributions: dict[str, torch.Tensor] = {}
        for term in self.terms:
            if isinstance(term, NonlinearTerm):
                contribution = term.contribution(state, time)
            else:
                contribution = term.multiplier * state[self.variable]
            if self.projection is not None:
                contribution = self.projection(contribution)
            contributions[term.name] = contribution
        return contributions


def merge_reads(
    reads: Iterable[Mapping[str, tuple[int, ...]]],
) -> dict[str, tuple[int, ...]]:
    """Return the union of several ``reads``, raising if two disagree on a key."""
    merged: dict[str, tuple[int, ...]] = {}
    for one in reads:
        for key, components in one.items():
            if merged.setdefault(key, components) != components:
                raise ValueError(
                    f"terms read {key!r} with component shapes "
                    f"{merged[key]} and {components}"
                )
    return merged


def _sum_multipliers(terms: Sequence[LinearTerm]) -> torch.Tensor:
    linear_operator = torch.zeros(())
    for term in terms:
        linear_operator = linear_operator + term.multiplier
    return linear_operator

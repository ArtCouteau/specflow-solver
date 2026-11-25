"""The evolution law of one variable: a sum of terms, optionally projected."""

from collections.abc import Callable, Sequence

import torch

from specflow.terms import LinearTerm, NonlinearTerm, State, Term

type Projection = Callable[[torch.Tensor], torch.Tensor]


class Equation:
    """The tendency of one variable: its linear operator plus its nonlinear tendency.

    Only the nonlinear tendency and the contributions are projected; the linear
    operator commutes with the projection.
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
        self.linear_operator = _sum_multipliers(self._linear_terms)

    def nonlinear_tendency(
        self, state: State, time: float, out_spectral: torch.Tensor
    ) -> None:
        """Write the projected sum of the nonlinear contributions into a tensor."""
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


def _sum_multipliers(terms: Sequence[LinearTerm]) -> torch.Tensor:
    linear_operator = torch.zeros(())
    for term in terms:
        linear_operator = linear_operator + term.multiplier
    return linear_operator

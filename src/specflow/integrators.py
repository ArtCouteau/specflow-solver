"""Fourth-order Runge-Kutta integrators over a sequence of equations.

An integrator starts a stepper on a state; the stepper owns the state's tensors and
three stage buffers per variable, allocated once, and advances the state in place.
"""

from collections.abc import Sequence
from typing import Protocol

import torch

from specflow.equation import Equation


class Stepper(Protocol):
    """Advances the state it owns, in place."""

    def step(self, time: float, time_step: float) -> None:
        """Advance the state from ``time`` to ``time + time_step``."""
        ...


class Integrator(Protocol):
    """The interface every integrator provides to the simulation."""

    def start(
        self, equations: Sequence[Equation], state: dict[str, torch.Tensor]
    ) -> Stepper:
        """Return a stepper that advances ``state``'s tensors in place."""
        ...


class RK4:
    """The classical RK4, applying the linear operator explicitly."""

    def start(
        self, equations: Sequence[Equation], state: dict[str, torch.Tensor]
    ) -> Stepper:
        """Return a stepper that advances ``state``'s tensors in place."""
        return _RungeKutta4Stepper(equations, state, integrating_factor=False)


class IFRK4:
    """The Lawson integrating-factor RK4, applying the linear operator exactly."""

    def start(
        self, equations: Sequence[Equation], state: dict[str, torch.Tensor]
    ) -> Stepper:
        """Return a stepper that advances ``state``'s tensors in place."""
        return _RungeKutta4Stepper(equations, state, integrating_factor=True)


class _RungeKutta4Stepper:
    def __init__(
        self,
        equations: Sequence[Equation],
        state: dict[str, torch.Tensor],
        integrating_factor: bool,
    ) -> None:
        self._equations = tuple(equations)
        self._state = state
        self._integrating_factor = integrating_factor
        variables = [equation.variable for equation in self._equations]
        self._variables = variables
        self._stage = {v: torch.empty_like(state[v]) for v in variables}
        self._slope = {v: torch.empty_like(state[v]) for v in variables}
        self._total = {v: torch.empty_like(state[v]) for v in variables}
        self._operators = {
            e.variable: e.linear_operator.to(state[e.variable]) for e in self._equations
        }
        ones = {v: state[v].new_ones(()) for v in variables}
        self._half_factors = ones
        self._full_factors = ones
        self._factor_time_step: float | None = None

    def step(self, time: float, time_step: float) -> None:
        self._update_factors(time_step)
        state, stage, slope, total = self._state, self._stage, self._slope, self._total
        half, full = self._half_factors, self._full_factors
        variables = self._variables

        self._evaluate(state, time)
        for v in variables:
            total[v].copy_(slope[v]).mul_(full[v])
            stage[v].copy_(slope[v]).mul_(time_step / 2).add_(state[v]).mul_(half[v])
        self._evaluate(stage, time + time_step / 2)
        for v in variables:
            total[v].addcmul_(half[v], slope[v], value=2)
            stage[v].copy_(state[v]).mul_(half[v]).add_(slope[v], alpha=time_step / 2)
        self._evaluate(stage, time + time_step / 2)
        for v in variables:
            total[v].addcmul_(half[v], slope[v], value=2)
            stage[v].copy_(slope[v]).mul_(half[v]).mul_(time_step)
            stage[v].addcmul_(full[v], state[v])
        self._evaluate(stage, time + time_step)
        for v in variables:
            total[v].add_(slope[v])
            state[v].mul_(full[v]).add_(total[v], alpha=time_step / 6)

    def _evaluate(self, state: dict[str, torch.Tensor], time: float) -> None:
        for equation in self._equations:
            variable = equation.variable
            slope = self._slope[variable]
            equation.nonlinear_tendency(state, time, slope)
            if not self._integrating_factor:
                slope.addcmul_(self._operators[variable], state[variable])

    def _update_factors(self, time_step: float) -> None:
        if not self._integrating_factor or time_step == self._factor_time_step:
            return
        self._half_factors = {
            v: operator.mul(time_step / 2).exp_()
            for v, operator in self._operators.items()
        }
        self._full_factors = {
            v: operator.mul(time_step).exp_() for v, operator in self._operators.items()
        }
        self._factor_time_step = time_step

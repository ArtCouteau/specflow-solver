"""Time-step controllers, which propose the time step of each step of a simulation.

The simulation shortens a step to land exactly on a stop time; a shortened step
never reaches the controller, so it never becomes the controller's time step.
"""

import math
from dataclasses import dataclass
from typing import Protocol, override

import torch

from specflow.mesh import Mesh
from specflow.terms import State


class TimeStepController(Protocol):
    """The interface every time-step controller provides to the simulation."""

    def reset(self) -> None:
        """Forget earlier proposals, so that the next one is a run's first."""
        ...

    def propose(self, mesh: Mesh, state: State) -> float:
        """Return the time step of the next step, from the current state."""
        ...

    @property
    def cfl_number(self) -> float | None:
        """The CFL number of the last proposal at its state, or None without one."""
        ...


@dataclass(frozen=True)
class FixedTimeStep(TimeStepController):
    """A prescribed, constant time step."""

    value: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.value) and self.value > 0):
            raise ValueError(f"time step must be positive and finite, got {self.value}")

    @override
    def reset(self) -> None:
        pass

    @override
    def propose(self, mesh: Mesh, state: State) -> float:
        return self.value

    @property
    @override
    def cfl_number(self) -> None:
        return None


class CFLTimeStep(TimeStepController):
    r"""A time step adapted to a target CFL number.

    The CFL number is $C = \Delta t \max \sum_i |u_i| / \Delta x$. The first
    proposal of every run is ``initial``, so a run resumed from a snapshot with its
    stored time step proposes what the uninterrupted run proposed. Each later
    proposal multiplies the previous one by ``target`` over its CFL number at the
    current state, capped at ``max_growth``.

    Args:
        variable: The state key of the velocity.
        target: The CFL number aimed at.
        initial: The first time step of every run.
        max_growth: The largest ratio between two successive time steps.
    """

    def __init__(
        self, variable: str, target: float, initial: float, max_growth: float = 1.2
    ) -> None:
        for name, value in (("target", target), ("initial time step", initial)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if not (math.isfinite(max_growth) and max_growth >= 1):
            raise ValueError(
                f"max growth must be finite and at least 1, got {max_growth}"
            )
        self.variable = variable
        self.target = target
        self.initial = initial
        self.max_growth = max_growth
        self._time_step = initial
        self._cfl_number: float | None = None

    @property
    @override
    def cfl_number(self) -> float | None:
        return self._cfl_number

    @override
    def reset(self) -> None:
        self._time_step = self.initial
        self._cfl_number = None

    @override
    def propose(self, mesh: Mesh, state: State) -> float:
        speed = self._speed(mesh, state[self.variable])
        if self._cfl_number is not None:
            cfl_number = self._time_step * speed
            if cfl_number * self.max_growth > self.target:
                self._time_step *= self.target / cfl_number
            else:
                self._time_step *= self.max_growth
        self._cfl_number = self._time_step * speed
        return self._time_step

    def _speed(self, mesh: Mesh, velocity_spectral: torch.Tensor) -> float:
        velocity_physical = mesh.decomposition.to_physical(velocity_spectral)
        speed_physical = velocity_physical.abs_().sum(dim=-mesh.dimension - 1)
        return mesh.decomposition.max(speed_physical).max().item() / mesh.spacing

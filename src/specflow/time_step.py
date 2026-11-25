"""Time-step controllers: the Δt of each step of a simulation.

The simulation shortens a step to land exactly on a stop time; a shortened step
never reaches the controller, so it never becomes the controller's Δt.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import torch

from specflow.mesh import Mesh
from specflow.terms import State


class TimeStepController(Protocol):
    """The interface every time-step controller provides to the simulation."""

    def propose(self, mesh: Mesh, state: State) -> float:
        """Return the controller's Δt for the next step from the current state."""
        ...

    @property
    def cfl_number(self) -> float | None:
        """The CFL number of the last proposal at its state, or None."""
        ...


@dataclass(frozen=True)
class FixedTimeStep:
    """A prescribed, constant Δt."""

    value: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.value) and self.value > 0):
            raise ValueError(f"time step must be positive and finite, got {self.value}")

    def propose(self, mesh: Mesh, state: State) -> float:
        """Return the prescribed Δt."""
        return self.value

    @property
    def cfl_number(self) -> None:
        """None, since a fixed Δt reads no velocity."""
        return None


class CFLTimeStep:
    """A Δt adapted to a target CFL number C = Δt·maxₓ Σᵢ|uᵢ|/Δx.

    The first proposal is the initial Δt, so a run resumed from a snapshot with its
    stored Δt proposes what the uninterrupted run proposed. Each later proposal is
    Δt·min(max_growth, target/C), with C from the previous proposal and the state.

    Args:
        variable: The state key of the velocity, (d,) in spectral space.
        target: The CFL number aimed at.
        initial: The first Δt.
        max_growth: The largest factor between two successive proposals.
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
        self.max_growth = max_growth
        self._time_step = initial
        self._cfl_number: float | None = None

    @property
    def cfl_number(self) -> float | None:
        """The CFL number of the last proposal at its state, None before the first."""
        return self._cfl_number

    def propose(self, mesh: Mesh, state: State) -> float:
        """Return the controller's Δt for the next step from the current state."""
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

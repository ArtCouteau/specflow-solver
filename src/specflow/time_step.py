"""Time-step controllers: the Δt of each step of a simulation.

The simulation shortens a step to land exactly on a stop time; a shortened step
never reaches the controller, so it never becomes the controller's Δt.
"""

import math
from dataclasses import dataclass
from typing import Protocol

from specflow.mesh import Mesh
from specflow.terms import State


class TimeStepController(Protocol):
    """The interface every time-step controller provides to the simulation."""

    def propose(self, mesh: Mesh, state: State) -> float:
        """Return the controller's Δt for the next step from the current state."""
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

"""The time integration of a set of equations from an initial state."""

from collections.abc import Sequence

import torch

from specflow.equation import Equation
from specflow.integrators import Integrator, Stepper
from specflow.mesh import Mesh
from specflow.terms import State
from specflow.time_step import TimeStepController

LANDING_TOLERANCE = 1e-10


class Simulation:
    """Equations advanced by an integrator, with Δt from a time-step controller."""

    def __init__(
        self,
        mesh: Mesh,
        equations: Sequence[Equation],
        integrator: Integrator,
        time_step: TimeStepController,
    ) -> None:
        self.mesh = mesh
        self.equations = tuple(equations)
        self.integrator = integrator
        self.time_step = time_step

    def run(self, state: State, start: float, end: float) -> dict[str, torch.Tensor]:
        """Return the state at ``end``, integrated from ``state`` at ``start``.

        The tensors of ``state`` are copied once and never modified; the run executes
        under inference mode, so the returned tensors are inference tensors.
        """
        with torch.inference_mode():
            owned = {key: tensor.clone() for key, tensor in state.items()}
            stepper = self.integrator.start(self.equations, owned)
            self._advance(stepper, owned, start, end)
        return owned

    def _advance(
        self, stepper: Stepper, state: State, time: float, stop: float
    ) -> float:
        while time < stop:
            time_step = self.time_step.propose(self.mesh, state)
            if stop - time <= time_step * (1 + LANDING_TOLERANCE):
                stepper.step(time, stop - time)
                time = stop
            else:
                stepper.step(time, time_step)
                time += time_step
        return time

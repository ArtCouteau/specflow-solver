"""The time integration of a set of equations from an initial state."""

import logging
import math
import time as clock
from collections.abc import Iterable, Iterator, Sequence

import torch

from specflow.equation import Equation, merge_reads
from specflow.integrators import Integrator, Stepper
from specflow.mesh import Mesh
from specflow.outputs import Outputs, OutputWriter
from specflow.terms import State
from specflow.time_step import TimeStepController

LANDING_TOLERANCE = 1e-10

logger = logging.getLogger(__name__)


class Simulation:
    """Equations advanced by an integrator, with time steps from a controller.

    With outputs, the output times are the start, every snapshot interval after it,
    and the end; steps are shortened to land on them exactly. The controller proposes
    a time step once per state reached, and a snapshot stores the time step proposed
    from its state.
    """

    def __init__(
        self,
        mesh: Mesh,
        equations: Sequence[Equation],
        integrator: Integrator,
        time_step: TimeStepController,
        outputs: Outputs | None = None,
    ) -> None:
        self.mesh = mesh
        self.equations = tuple(equations)
        self.integrator = integrator
        self.time_step = time_step
        self.outputs = outputs
        self._components = self._check_consistency()

    def run(self, state: State, start: float, end: float) -> dict[str, torch.Tensor]:
        """Return the state at ``end``, integrated from ``state`` at ``start``.

        The tensors of ``state`` are copied once and never modified; the run executes
        under inference mode, so the returned tensors are inference tensors.
        """
        if not end >= start:
            raise ValueError(f"end {end} is before start {start}")
        self._check_state(state)
        self.time_step.reset()
        with torch.inference_mode():
            owned = {key: tensor.clone() for key, tensor in state.items()}
            stepper = self.integrator.start(self.equations, owned)
            if self.outputs is None:
                self._integrate(stepper, owned, start, [end], None)
            else:
                writer = self.outputs.open(self.mesh, start)
                stops = _output_times(
                    start, end, self.outputs.snapshot_interval, writer.resumed
                )
                self._integrate(stepper, owned, start, stops, writer)
        return owned

    def _check_consistency(self) -> dict[str, tuple[int, ...] | None]:
        variables: list[str] = []
        for equation in self.equations:
            if equation.variable in variables:
                raise ValueError(
                    f"variable {equation.variable!r} evolved by more than one equation"
                )
            variables.append(equation.variable)
        reads = merge_reads(equation.reads for equation in self.equations)
        for key in reads:
            if key not in variables:
                raise ValueError(f"terms read {key!r}, which no equation evolves")
        return {variable: reads.get(variable) for variable in variables}

    def _check_state(self, state: State) -> None:
        for variable in self._components:
            if variable not in state:
                raise ValueError(f"state is missing the variable {variable!r}")
        for key, tensor in state.items():
            if key not in self._components:
                raise ValueError(f"state key {key!r} is evolved by no equation")
            self._check_tensor(key, tensor)
        for equation in self.equations:
            equation.check_projected(self.mesh, state)

    def _check_tensor(self, key: str, tensor: torch.Tensor) -> None:
        mesh = self.mesh
        components = self._components[key]
        if components is None:
            expected = mesh.spectral_shape
        else:
            expected = components + mesh.spectral_shape
        if tensor.ndim < len(expected) or tensor.shape[-len(expected) :] != expected:
            raise ValueError(
                f"state key {key!r} has shape {tuple(tensor.shape)}, "
                f"expected (..., {', '.join(map(str, expected))})"
            )
        if tensor.dtype != mesh.complex_dtype:
            raise ValueError(
                f"state key {key!r} has dtype {tensor.dtype}, "
                f"expected {mesh.complex_dtype}"
            )
        if tensor.device != mesh.wavenumbers.device:
            raise ValueError(
                f"state key {key!r} is on device {tensor.device}, "
                f"expected {mesh.wavenumbers.device}"
            )

    def _integrate(
        self,
        stepper: Stepper,
        state: State,
        start: float,
        stops: Iterable[float],
        writer: OutputWriter | None,
    ) -> None:
        time = start
        step = 0 if writer is None else writer.step
        proposed = self._propose(state, time, step)
        wall_start, step_start = clock.perf_counter(), step
        if writer is not None and not writer.resumed:
            writer.sample(state, time, step)
        for stop in stops:
            while time < stop:
                if stop - time <= proposed * (1 + LANDING_TOLERANCE):
                    taken = stop - time
                    stepper.step(time, taken)
                    time = stop
                else:
                    taken = proposed
                    stepper.step(time, taken)
                    time += taken
                step += 1
                logger.debug(f"step {step}, time {time:.6g}, time_step {taken:.6g}")
                if writer is not None:
                    writer.sample(state, time, step)
                proposed = self._propose(state, time, step)
            if writer is not None:
                writer.write(state, time, step, proposed)
                wall = clock.perf_counter()
                cfl_number = self.time_step.cfl_number
                cfl = "" if cfl_number is None else f"CFL {cfl_number:.3g}, "
                per_step = (wall - wall_start) / max(step - step_start, 1)
                logger.info(
                    f"step {step}, time {time:.6g}, time_step {proposed:.6g}, "
                    f"{cfl}{per_step:.3g} s per step"
                )
                wall_start, step_start = wall, step

    def _propose(self, state: State, time: float, step: int) -> float:
        proposed = self.time_step.propose(self.mesh, state)
        cfl_number = self.time_step.cfl_number
        if cfl_number is not None and not math.isfinite(cfl_number):
            raise FloatingPointError(
                f"CFL number is not finite at step {step}, time {time:.6g}"
            )
        return proposed


def _output_times(
    start: float, end: float, interval: float, resumed: bool
) -> Iterator[float]:
    index = 1 if resumed else 0
    while end - (start + index * interval) > LANDING_TOLERANCE * interval:
        yield start + index * interval
        index += 1
    if end > start or not resumed:
        yield end

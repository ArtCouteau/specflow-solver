"""The time integration of a set of equations from an initial state."""

from collections.abc import Sequence

import torch

from specflow.equation import Equation, merge_reads
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
        self._components = self._check_consistency()

    def run(self, state: State, start: float, end: float) -> dict[str, torch.Tensor]:
        """Return the state at ``end``, integrated from ``state`` at ``start``.

        The tensors of ``state`` are copied once and never modified; the run executes
        under inference mode, so the returned tensors are inference tensors.
        """
        if not end >= start:
            raise ValueError(f"end {end} is before start {start}")
        self._check_state(state)
        with torch.inference_mode():
            owned = {key: tensor.clone() for key, tensor in state.items()}
            stepper = self.integrator.start(self.equations, owned)
            self._advance(stepper, owned, start, end)
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

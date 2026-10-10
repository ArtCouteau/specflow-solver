from __future__ import annotations

import time
from dataclasses import dataclass

from specflow.core import defs
from specflow.core.numerics.backend.backend import NumericsBackend
from specflow.core.physics.wrappers.wrapper import Scalar, Vector
from specflow.simulation.control import SimulationControls
from specflow.simulation.equation import Equation
from specflow.simulation.integration import TimeIntegrator


@dataclass(frozen=True)
class Solver:
    control: SimulationControls
    integrator: TimeIntegrator
    equations: list[Equation[Scalar | Vector]]
    backend: NumericsBackend

    def _step(self):
        self.integrator.advance(self.equations, self.control.time.current)

        for eq in self.equations:
            if eq.variable.data.isnan().any():
                raise ValueError(f"NaN values detected in the variable of equation {eq.tag}")

    def run(self) -> None:
        print("\n\nStarting time loop\n")

        self.backend.logger.info("Clean output directory")
        self.backend.io.output.make_dir("output")

        self.backend.logger.info("Running time loop")
        # self._write_state()

        _t0 = time.perf_counter()
        while self.control.time.advance():
            self.backend.logger.info(f"\nStep {self.control.time.step}")
            self.backend.logger.info(f"  Time = {self.backend.io.output.timestamp(self.control.time.current)}")

            # self.integrator.dt = self.control.time.clamp_dt()
            self.backend.logger.info(f"  Time step = {self.integrator.dt:.{defs.TIMESTAMP_DIGITS}f}")

            self._step()

            self.control.time.update(self.integrator.dt)
            if self.control.time.save.active:
                # self._write_state()
                self.control.time.save.deactivate()

        _t1 = time.perf_counter()
        self.backend.logger.info("\nSimulation has successfully finished")
        self.backend.logger.info(f"Elapsed time: {(_t1 - _t0) / 3600:.2f} hours")
        self.backend.logger.info(f"Average time step duration: {(_t1 - _t0) / self.control.time.step:.3f} seconds")

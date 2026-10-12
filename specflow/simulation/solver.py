from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import specflow.processing.physics  # noqa: F401  registers the post-processing rules
from specflow.core import defs
from specflow.core.numerics.backend.backend import NumericsBackend
from specflow.core.physics.wrappers.constant import Constant
from specflow.core.physics.wrappers.wrapper import Scalar, Vector
from specflow.processing.numerics.kind import kind_tag
from specflow.processing.numerics.node import Node
from specflow.processing.numerics.operators import OPERATORS
from specflow.processing.numerics.processor import PostProcessor
from specflow.processing.numerics.rule import RULES, Rule
from specflow.simulation.control import SimulationControls
from specflow.simulation.equation import TIME, Equation
from specflow.simulation.integration import TimeIntegrator

_TAG = re.compile(r"[A-Z][A-Za-z0-9]*")


@dataclass(frozen=True)
class Solver:
    control: SimulationControls
    integrator: TimeIntegrator
    equations: list[Equation[Scalar | Vector]]
    backend: NumericsBackend
    statistics: tuple[Node[Any], ...] = ()
    rules: tuple[Rule[Any], ...] = ()
    postprocessor: PostProcessor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for tag in (tag for eq in self.equations for tag in (eq.tag, *(term.tag for term in eq.terms.values()))):
            if not _TAG.fullmatch(tag):
                raise ValueError(f"Equation and term tags must be PascalCase, got '{tag}'")

        postprocessor = PostProcessor(
            rules=[*(rule for eq in self.equations for rule in eq.rules()), *self.rules, *RULES],
            operators=OPERATORS,
            inputs=[*(Node(eq.tag, eq.variable.kind) for eq in self.equations), TIME],
            targets=self.statistics,
        )
        object.__setattr__(self, "postprocessor", postprocessor)
        for step in postprocessor.steps:
            self.backend.logger.debug(f"Post-processing step: {step}")

    def _write_statistics(self) -> None:
        if not self.statistics:
            return

        mesh = self.equations[0].variable.mesh
        inputs = {Node(eq.tag, eq.variable.kind): eq.variable for eq in self.equations}
        inputs[TIME] = Constant.scalar(self.control.time.current, self.backend, mesh)

        folder = f"output/{self.backend.io.output.timestamp(self.control.time.current)}"
        self.backend.io.output.make_dir(folder)
        for node, value in self.postprocessor.run(inputs).items():
            self.backend.io.h5.write_tensor(f"{folder}/statistics", f"{node.name}/{kind_tag(node.kind)}", value.data)

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
        self._write_statistics()

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
                self._write_statistics()
                self.control.time.save.deactivate()

        _t1 = time.perf_counter()
        self.backend.logger.info("\nSimulation has successfully finished")
        self.backend.logger.info(f"Elapsed time: {(_t1 - _t0) / 3600:.2f} hours")
        self.backend.logger.info(f"Average time step duration: {(_t1 - _t0) / self.control.time.step:.3f} seconds")

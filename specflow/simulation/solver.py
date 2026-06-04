from __future__ import annotations

import time

from . import control, equation, integration, solution
from ..core import defs
from ..core.numerics import context


class Solver:
    control: control.SimulationControls
    integrator: integration.TimeIntegrator
    equations: dict[str, equation.Equation]
    variable: solution.SolutionVariable

    def __init__(self):
        context.get().generate_physics_context()
        self.integrator = integration.TimeIntegratorRegistry.get(context.get().config.get('schemes/integration', str))()
        self.equations = {tag: equation.Equation(self, tag)
                          for tag in context.get().config.get('equations', dict).keys()}
        self.variable = solution.SolutionVariable({tag: eq.variable.data for tag, eq in self.equations.items()})

    def _step(self):
        self.integrator.advance()

        for key, tensor in self.variable.data.items():
            if tensor.isnan().any():
                raise ValueError(f'NaN values detected in the variable of equation {key}')

    def _write_state(self):
        context.get().io.output.make_dir(f'output/{self.control.time.timestamp()}')
        for equation in self.equations.values():
            equation.write_state()

    def compute_rhs(self, variable: solution.SolutionVariable, time: float) -> solution.SolutionVariable:
        return solution.SolutionVariable({
            key: self.equations[key].compute_rhs(variable, time)
            for key in self.equations
        })

    def run(self) -> None:
        print('\n\nStarting time loop\n')

        print('Clean output directory')
        context.get().io.output.make_or_clean_dir('output')

        print('Running time loop')
        _t0 = time.perf_counter()
        while self.control.time.advance():
            print(f'Step {self.control.time.step}')
            print(f'Time = {self.control.time.timestamp()}')

            self.integrator.update()
            print(f'Time step = {self.integrator.dt:.{defs.TIMESTAMP_DIGITS}f}')

            if self.control.time.save.active:
                self._write_state()
                self.control.time.save.deactivate()

            self._step()

            self.control.time.update()

        _t1 = time.perf_counter()
        print(f'Simulation has succefully finished')
        print(f'Elapsed time: {(_t1 - _t0)/3600:.2f} hours')
        print(f'Average time step duration: {(_t1 - _t0)/self.control.time.step:.3f} seconds')

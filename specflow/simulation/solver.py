from __future__ import annotations

import time

from . import equation
from ..core.numerics import context


class Solver:
    equations: dict[str, equation.Equation]

    def __init__(self):
        context.get().generate_simulation_context()
        context.get().generate_physics_context()
        self.equations = {tag: equation.Equation(tag) for tag in context.get().config.get('equations', dict).keys()}

    def _setup(self):
        context.get().sim.integrator.setup(self.equations)

    def _step(self):
        for equation in self.equations.values():
            equation.step()

    def _write_state(self):
        context.get().io.output.make_dir(f'output/{context.get().sim.control.time.timestamp()}')
        for equation in self.equations.values():
            equation.write_state()

    def run(self) -> None:
        print('\n\nStarting time loop\n')

        print('Clean output directory')
        context.get().io.output.make_or_clean_dir('output')

        print('Setup simulation')
        self._setup()

        print('Running time loop')
        _t0 = time.perf_counter()
        _ctrl = context.get().sim.control
        while _ctrl.time.advance():
            print(f'Step {_ctrl.time.step}')
            print(f'Time = {_ctrl.time.current:.5f}')

            context.get().sim.integrator.update()
            print(f'Time step = {context.get().sim.integrator.dt:.5f}')

            if _ctrl.time.save.active:
                self._write_state()

            self._step()

            _ctrl.time.update()

        _t1 = time.perf_counter()
        print(f'Simulation has succefully finished')
        print(f'Elapsed time: {(_t1 - _t0)/3600:.2f} hours')
        print(f'Average time step duration: {(_t1 - _t0)/_ctrl.time.step:.3f} seconds')

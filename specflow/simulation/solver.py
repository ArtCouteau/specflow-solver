from __future__ import annotations

import time

from . import equation
from ..core.numerics import get_context


class Solver:
    equations: dict[str, equation.Equation]

    def __init__(self):
        get_context().generate_simulation_context()
        get_context().generate_physics_context()
        self.equations = {tag: equation.Equation(tag) for tag in get_context().config.get('equations', dict).keys()}

    def _step(self):
        for equation in self.equations.values():
            equation.step()

    def run(self) -> None:
        print('\n\nStarting time loop\n')

        print('Clean output directory')
        get_context().io.output.make_or_clean_dir('output')

        _t0 = time.perf_counter()
        _ctrl = get_context().sim.control
        while _ctrl.time.advance():
            print(f'Step {_ctrl.time.step}')
            print(f'Time = {_ctrl.time.current:.5f}')

            get_context().sim.integrator.update()
            print(f'Time step = {get_context().sim.integrator.dt:.5f}')

            if _ctrl.time.save.active:
                get_context().io.output.make_dir(f'output/{_ctrl.time.timestamp()}')

            self._step()

            _ctrl.time.update()

        _t1 = time.perf_counter()
        print(f'Simulation has succefully finished')
        print(f'Elapsed time: {(_t1 - _t0)/3600:.2f} hours')
        print(f'Average time step duration: {(_t1 - _t0)/_ctrl.time.step:.3f} seconds')

from __future__ import annotations

import time
import torch
from typing import Literal

from . import control, equation, integration, solution
from ..core import defs
from ..core import context
from ..core.numerics.operations import fourier
from ..core.physics import variable


class Solver:
    control: control.SimulationControls
    integrator: integration.TimeIntegrator
    equations: dict[str, equation.Equation]
    variable: solution.SolutionVariable

    def __init__(self):
        self.control = control.SimulationControls(self)
        self.integrator = integration.TimeIntegratorRegistry.get(
            context.get().config.get('schemes/integration', str)
        )(self)
        self.equations = {
            key: equation.EquationRegistry.get(
                context.get().config.get(f'equations/{key}/prebuild', str, 'custom')
            )(key)
            for key in context.get().config.get('equations', dict).keys()
        }
        self._init_variable()

    def _init_variable(self):
        self.variable = solution.SolutionVariable({
            eq.variable: context.get().memory.alloc(
                f'solution/{eq.variable}',
                variable.field_shape(eq.rank),
                dtype=context.get().config.get('numerics/precision/complex', torch.dtype),
                device=context.get().config.get('numerics/device', str)
            ) for eq in self.equations.values()
        })
        self.variable.zero_()

    def _add_time_step_limiters(self):
        for eq in self.equations.values():
            for term in eq.terms.values():
                if (limiter := term.time_step_limiter()) is not None:
                    self.integrator.add_time_step_limiter(*limiter)

    def _setup(self):
        self.integrator.setup()

    def _step(self):
        self.integrator.advance()

        for key, tensor in self.variable.data.items():
            if tensor.isnan().any():
                raise ValueError(f'NaN values detected in the variable of equation {key}')

    def _write_state(self):
        timestamp = context.get().io.output.timestamp(self.control.time.current)
        path = f'output/{timestamp}/simulation'
        context.get().io.output.make_dir(f'output/{timestamp}')
        for eq in self.equations.values():
            context.get().io.h5.write_tensor(
                path,
                f'fields/{eq.variable}',
                self.variable.get(eq.variable),
                {'rank': eq.rank.name}
            )

            for term in eq.terms.values():
                for key, value in term.restart_state().items():
                    context.get().io.h5.write_tensor(
                        path,
                        f'restart/{term.tag}/{key}',
                        value,
                        {'rank': eq.rank.name}
                    )

                for key, value in term.diagnostic_fields(self.variable, self.control.time.current).items():
                    context.get().io.h5.write_tensor(
                        path,
                        f'fields/{key}',
                        value
                    )

    def set_initial_condition(self, variable: str, data: torch.Tensor, space: Literal['physical', 'spectral']):
        if space == 'physical':
            self.variable.get(variable).copy_(fourier.to_spectral(data))
        else:
            self.variable.get(variable).copy_(data)

    def run(self) -> None:
        print('\n\nStarting time loop\n')

        print('Clean output directory')
        context.get().io.output.make_or_clean_dir('output')

        self._setup()

        print('Running time loop')
        self._write_state()

        _t0 = time.perf_counter()
        while self.control.time.advance():
            print(f'Step {self.control.time.step}')
            print(f'Time = {context.get().io.output.timestamp(self.control.time.current)}')

            self.integrator.update(self.variable)
            self.control.time.clamp_dt()
            print(f'Time step = {self.integrator.dt:.{defs.TIMESTAMP_DIGITS}f}')

            self._step()
            self.control.time.update()

            if self.control.time.save.active:
                self._write_state()
                self.control.time.save.deactivate()

        _t1 = time.perf_counter()
        print(f'Simulation has successfully finished')
        print(f'Elapsed time: {(_t1 - _t0)/3600:.2f} hours')
        print(f'Average time step duration: {(_t1 - _t0)/self.control.time.step:.3f} seconds')

from __future__ import annotations

from typing import Callable

from . import solution, solver
from ..core import context, defs
from ..core.registry import Registry


class TimeIntegrator:
    dt: float
    limiters: dict[str, TimeStepLimiter]
    solver: solver.Solver

    def __init__(self, solver: solver.Solver):
        self.solver = solver
        self.dt = context.get().config.get('time/dt', float)
        self.limiters = {}

    def _compute_time_derivative(self) -> solution.SolutionVariable: ...

    def _compute_rhs(self, variable: solution.SolutionVariable, time: float) -> solution.SolutionVariable:
        return solution.SolutionVariable({
            eq.variable: eq.compute_rhs(variable, time)
            for eq in self.solver.equations.values()
        })

    def _get_time_step_target_ratio(self, variable: solution.SolutionVariable) -> float:
        _ratio = 1.25
        for key, limiter in self.limiters.items():
            _value = limiter.current(variable, self.dt)
            context.get().log.info(f'Time step limiter: {key} = {_value:.5f}')
            _ratio = min(_ratio, limiter.target/(_value + defs.SMALL))

        return max(0.8, min(1.25, _ratio))

    def setup(self): ...

    def add_time_step_limiter(
            self,
            tag: str,
            target: float,
            func: Callable[[solution.SolutionVariable, float], float]
    ) -> None:
        if tag not in self.limiters:
            self.limiters[tag] = TimeStepLimiter(tag, target, func)
        else:
            context.get().log.warning(f'Time step limiter for "{tag}" already exists. Skipping')

        print(f'Added time step limiter "{tag}" (target = {target:.2f})')

    def update(self, variable: solution.SolutionVariable) -> None:
        self.dt *= self._get_time_step_target_ratio(variable)

    def overwrite_dt(self, dt: float) -> None:
        self.dt = dt

    def advance(self) -> None:
        self.solver.variable.add_(self.dt * self._compute_time_derivative())


class TimeIntegratorRegistry(Registry[TimeIntegrator]):
    pass


@TimeIntegratorRegistry.register('EulerForward')
class EulerForward(TimeIntegrator):
    def _compute_time_derivative(self) -> solution.SolutionVariable:
        return self._compute_rhs(self.solver.variable, self.solver.control.time.current)


@TimeIntegratorRegistry.register('RungeKutta4')
class RungeKutta4(TimeIntegrator):
    k: list[solution.SolutionVariable]
    stage: solution.SolutionVariable

    def __init__(self, solver: solver.Solver):
        super().__init__(solver)

    def setup(self):
        self.k = [
            solution.SolutionVariable({
                key: context.get().memory.alloc(
                    f'RungeKutta4/k{i}/{key}',
                    tensor.shape,
                    dtype=tensor.dtype,
                    device=tensor.device,
                ) for key, tensor in self.solver.variable.data.items()
            })
            for i in range(4)
        ]
        [self.k[i].zero_() for i in range(4)]

        self.stage = solution.SolutionVariable({
            key: context.get().memory.alloc(
                f'RungeKutta4/stage/{key}',
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
            ) for key, tensor in self.solver.variable.data.items()
        })
        self.stage.zero_()

    def _compute_time_derivative(self) -> solution.SolutionVariable:
        k = self.k
        stage = self.stage
        t = self.solver.control.time.current
        u = self.solver.variable

        k[0].copy_(self._compute_rhs(u, t))

        stage.copy_(u).add_(k[0], alpha=0.5*self.dt)
        k[1].copy_(self._compute_rhs(stage, t + 0.5*self.dt))

        stage.copy_(u).add_(k[1], alpha=0.5*self.dt)
        k[2].copy_(self._compute_rhs(stage, t + 0.5*self.dt))

        stage.copy_(u).add_(k[2], alpha=self.dt)
        k[3].copy_(self._compute_rhs(stage, t + self.dt))

        return (k[0] + 2*k[1] + 2*k[2] + k[3]) / 6


@TimeIntegratorRegistry.register('RungeKutta4LowStorage')
class RungeKutta4LowStorage(TimeIntegrator):
    _alpha = [0.0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748]
    _beta = [0.0, -0.4178904745,   -1.192151694643, -1.697784692471, -1.514183444257]
    _gamma = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]
    h: solution.SolutionVariable

    def __init__(self, solver: solver.Solver):
        super().__init__(solver)

    def setup(self):
        self.h = solution.SolutionVariable({
            key: context.get().memory.alloc(
                f'RungeKutta4LowStorage/h/{key}',
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
            ) for key, tensor in self.solver.variable.data.items()
        })
        self.h.zero_()

    def advance(self) -> None:
        h = self.h
        h.zero_()
        u = self.solver.variable
        t = self.solver.control.time.current

        for i in range(5):
            h.mul_(self._beta[i]).add_(self._compute_rhs(u, t + self._alpha[i] * self.dt))
            u.add_(self._gamma[i] * self.dt * h)


class TimeStepLimiter(object):
    tag: str
    target: float
    _current: Callable[[solution.SolutionVariable, float], float]

    def __init__(self, tag: str, target: float, func: Callable[[solution.SolutionVariable, float], float]) -> None:
        self.tag = tag
        self.target = target
        self._current = func

    def current(self, variable: solution.SolutionVariable, dt: float) -> float:
        return self._current(variable, dt)

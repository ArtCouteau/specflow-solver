from __future__ import annotations

from typing import Callable, Type

from . import solution, solver
from ..core import defs
from ..core.numerics import context


class TimeIntegratorRegistry:
    _registry: dict[str, Type[TimeIntegrator]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(integrator_class: Type[TimeIntegrator]) -> Type[TimeIntegrator]:
            cls._registry[name] = integrator_class
            return integrator_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[TimeIntegrator]:
        if name not in cls._registry:
            raise ValueError(
                f'Time integration model {name} not found, available models:\n'
                + '\n'.join(f"- {k}" for k in cls._registry.keys())
            )
        return cls._registry[name]


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

    def _get_time_step_target_ratio(self) -> float:
        _ratio = 1.25
        for key, limiter in self.limiters.items():
            _value = limiter.current()
            print(f'Time step limiter: {key} = {_value:.5f}')
            _ratio = min(_ratio, limiter.target/(_value + defs.SMALL))

        return max(0.8, min(1.25, _ratio))

    def setup(self): ...

    def add_time_step_limiter(self, tag: str, target: float, func: Callable[[], float]) -> None:
        if not tag in self.limiters.keys():
            self.limiters[tag] = TimeStepLimiter(tag, target, func)
        else:
            print(f'Time step limiter for "{tag}" already exists. Skipping')

        print(f'Added time step limiter "{tag}" (target = {target:.2f})')

    def update(self) -> None:
        self.dt *= self._get_time_step_target_ratio()

    def overwrite_dt(self, dt: float) -> None:
        self.dt = dt

    def advance(self) -> None:
        self.solver.variable.add_(self.dt * self._compute_time_derivative())


@TimeIntegratorRegistry.register('EulerForward')
class EulerForward(TimeIntegrator):
    def _compute_time_derivative(self) -> solution.SolutionVariable:
        return self._compute_rhs(self.solver.variable, self.solver.control.time.current)


@TimeIntegratorRegistry.register('RungeKutta4')
class RungeKutta4(TimeIntegrator):
    k: list[solution.SolutionVariable]

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

    def _compute_time_derivative(self) -> solution.SolutionVariable:
        k = self.k
        t = self.solver.control.time.current
        u = self.solver.variable

        k[0].copy_(self._compute_rhs(u,                    t))
        k[1].copy_(self._compute_rhs(u + 0.5*self.dt*k[0], t + 0.5*self.dt))
        k[2].copy_(self._compute_rhs(u + 0.5*self.dt*k[1], t + 0.5*self.dt))
        k[3].copy_(self._compute_rhs(u + self.dt*k[2],     t + self.dt))

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


@TimeIntegratorRegistry.register('AdamsBashforth4')
class AdamsBashforth4(TimeIntegrator):
    _weights = [
        [1.],
        [3/2,   -1/2],
        [23/12, -4/3,   5/12],
        [55/24, -59/24, 37/24, -3/8],
    ]

    k: list[solution.SolutionVariable]
    _step: int

    def __init__(self, solver: solver.Solver):
        super().__init__(solver)

    def setup(self):
        self.k = [
            solution.SolutionVariable({
                key: context.get().memory.alloc(
                    f'AdamsBashforth4/k{i}/{key}',
                    tensor.shape,
                    dtype=tensor.dtype,
                    device=tensor.device,
                ) for key, tensor in self.solver.variable.data.items()
            })
            for i in range(3)
        ]
        self._step = 0

    def _compute_time_derivative(self) -> solution.SolutionVariable:
        k = self.k
        w = self._weights[min(self._step, 3)]
        t = self.solver.control.time.current
        u = self.solver.variable

        f_n = self._compute_rhs(u, t)

        result = w[0] * f_n
        for j, wj in enumerate(w[1:]):
            result = result + wj * k[j]

        for j in range(min(self._step, 2), 0, -1):
            k[j].copy_(k[j - 1])
        k[0].copy_(f_n)
        self._step += 1

        return result


class TimeStepLimiter(object):
    tag: str
    target: float
    _current: Callable[[], float]

    def __init__(self, tag: str, target: float, func: Callable[[], float]) -> None:
        self.tag = tag
        self.target = target
        self._current = func

    def current(self) -> float:
        return self._current()

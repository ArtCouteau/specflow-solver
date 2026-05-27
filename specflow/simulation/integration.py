from __future__ import annotations
from typing import Callable, Type

import torch

from . import equation
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

    def __init__(self):
        self.dt = context.get().config.get('time/dt', float)
        self.limiters = {}

    def _compute_time_derivative(self, equation: equation.Equation) -> torch.Tensor: ...

    def add_time_step_limiter(self, tag: str, target: float, func: Callable[[], float]) -> None:
        if not tag in self.limiters.keys():
            self.limiters[tag] = TimeStepLimiter(tag, target, func)
        else:
            raise KeyError(f'Time step limiter for "{tag}" already exists.')

    def update(self) -> None:
        # Add time step limiters logic
        self.dt = context.get().config.get('time/dt', float)

    def overwrite_dt(self, dt: float) -> None:
        self.dt = dt

    def advance(self, equation: equation.Equation) -> None:
        equation.variable.data.add_(self.dt*self._compute_time_derivative(equation))

    def setup(self, equations: dict[str, equation.Equation]) -> None:
        pass


@TimeIntegratorRegistry.register('EF')
class EulerForward(TimeIntegrator):
    def _compute_time_derivative(self, equation: equation.Equation) -> torch.Tensor:
        return equation.compute_rhs(equation.variable.data, context.get().sim.control.time.current)


@TimeIntegratorRegistry.register('RK4')
class RungeKutta4(TimeIntegrator):
    k: dict[str, list[torch.Tensor]]

    def setup(self, equations: dict[str, equation.Equation]) -> None:
        self.k = {
            tag: [
                context.get().memory.alloc(
                    f'RungeKutta/{tag}/k{i}',
                    equation.variable.data.shape,
                    equation.variable.data.dtype,
                    equation.variable.data.device
                )
                for i in range(4)
            ]
            for tag, equation in equations.items()
        }

    def _compute_time_derivative(self, equation: equation.Equation) -> torch.Tensor:
        k = self.k[equation.tag]
        t = context.get().sim.control.time.current
        u = equation.variable.data

        k[0].copy_(equation.compute_rhs(u,                    t))
        k[1].copy_(equation.compute_rhs(u + 0.5*self.dt*k[0], t + 0.5*self.dt))
        k[2].copy_(equation.compute_rhs(u + 0.5*self.dt*k[1], t + 0.5*self.dt))
        k[3].copy_(equation.compute_rhs(u + self.dt*k[2],     t + self.dt))

        return (k[0] + 2*k[1] + 2*k[2] + k[3]) / 6


@TimeIntegratorRegistry.register('RK4LS')
class RungeKutta4LowStorage(TimeIntegrator):
    _alpha = [0.0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748]
    _beta = [0.0, -0.4178904745,   -1.192151694643, -1.697784692471, -1.514183444257]
    _gamma = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]

    h: dict[str, torch.Tensor]

    def setup(self, equations: dict[str, equation.Equation]) -> None:
        self.h = {
            tag: context.get().memory.alloc(
                f'RungeKutta4LowStorage/{tag}/h',
                eq.variable.data.shape,
                eq.variable.data.dtype,
                eq.variable.data.device
            )
            for tag, eq in equations.items()
        }

    def advance(self, equation: equation.Equation) -> None:
        h = self.h[equation.tag]
        h.zero_()
        u = equation.variable.data
        t = context.get().sim.control.time.current

        for i in range(5):
            h.mul_(self._beta[i]).add_(equation.compute_rhs(u, t + self._alpha[i] * self.dt))
            u.add_(self._gamma[i] * self.dt * h)


@TimeIntegratorRegistry.register('AB4')
class AdamsBashforth4(TimeIntegrator):
    _weights = [
        [1.],
        [3/2,   -1/2],
        [23/12, -4/3,   5/12],
        [55/24, -59/24, 37/24, -3/8],
    ]

    _rhs:  dict[str, list[torch.Tensor]]
    _step: dict[str, int]

    def setup(self, equations: dict[str, equation.Equation]) -> None:
        self._rhs = {
            tag: [
                context.get().memory.alloc(
                    f'AdamsBashforth4/{tag}/rhs{i}',
                    eq.variable.data.shape,
                    eq.variable.data.dtype,
                    eq.variable.data.device
                )
                for i in range(3)
            ]
            for tag, eq in equations.items()
        }
        self._step = {tag: 0 for tag in equations}

    def advance(self, equation: equation.Equation) -> None:
        rhs = self._rhs[equation.tag]
        step = self._step[equation.tag]
        w = self._weights[min(step, 3)]
        f_n = equation.compute_rhs(equation.variable.data, context.get().sim.control.time.current)

        equation.variable.data.add_(self.dt * w[0] * f_n)
        for j, wj in enumerate(w[1:]):
            equation.variable.data.add_(self.dt * wj * rhs[j])

        for j in range(min(step, 2), 0, -1):
            rhs[j].copy_(rhs[j - 1])
        rhs[0].copy_(f_n)
        self._step[equation.tag] += 1


class TimeStepLimiter(object):
    tag: str
    target: float
    current: Callable[[], float]

    def __init__(self, tag: str, target: float, func: Callable[[], float]) -> None:
        self.tag = tag
        self.target = target
        self.current = func

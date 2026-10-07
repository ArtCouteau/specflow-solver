from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, override

import torch

from specflow.core.numerics.operations import fourier
from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Scalar, Spectral, Vector
from specflow.simulation.equation import Equation, TRankEquation


@dataclass
class TimeIntegrator:
    dt: float
    limiters: dict[str, TimeStepLimiter]
    _buffers: dict[str, dict[str, Field[Scalar | Vector, Spectral]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _buffer(self, name: str, eq: Equation[TRankEquation]) -> Field[TRankEquation, Spectral]:
        slot = self._buffers.setdefault(eq.tag, {})
        if name not in slot:
            slot[name] = eq.variable.replace(data=torch.zeros_like(eq.variable.data))

        if slot[name].rank is eq.rank:
            return slot[name]
        else:
            raise TypeError(f"Buffer holds a field of rank {slot[name].rank} not equal to equation rank {eq.rank}")

    def _compute_rhs(self, equations: list[Equation[TRankEquation]], time: float) -> None:
        for eq in equations:
            eq.compute_rhs(time)

    @abstractmethod
    def advance(self, equations: list[Equation[TRankEquation]], time: float) -> None: ...


class EulerForward(TimeIntegrator):
    @override
    def advance(self, equations: list[Equation[TRankEquation]], time: float) -> None:
        self._compute_rhs(equations, time)
        for eq in equations:
            eq.variable.add_(eq.rhs, alpha=self.dt)


class RungeKutta4(TimeIntegrator):
    @override
    def advance(self, equations: list[Equation[TRankEquation]], time: float) -> None:
        for eq in equations:
            self._buffer("u0", eq).copy_(eq.variable)
            self._buffer("accumulating", eq).zero_()
        for c, w, tau in ((0.5, 1, 0.0), (0.5, 2, 0.5), (1.0, 2, 0.5), (None, 1, 1.0)):
            self._compute_rhs(equations, time + tau * self.dt)
            for eq in equations:
                accumulating = self._buffer("accumulating", eq).add_(eq.rhs, alpha=w)
                if c is not None:
                    eq.variable.copy_(self._buffer("u0", eq)).add_(eq.rhs, alpha=c * self.dt)
                else:
                    eq.variable.copy_(self._buffer("u0", eq)).add_(accumulating, alpha=self.dt / 6)


class RungeKutta4LowStorage(TimeIntegrator):
    _alpha: ClassVar[tuple[float, ...]] = (0.0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748)
    _beta: ClassVar[tuple[float, ...]] = (0.0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257)
    _gamma: ClassVar[tuple[float, ...]] = (
        0.1496590219993,
        0.3792103129999,
        0.8229550293869,
        0.6994504559488,
        0.1530572479681,
    )

    @override
    def advance(self, equations: list[Equation[TRankEquation]], time: float) -> None:
        for eq in equations:
            self._buffer("h", eq).zero_()
            for a, b, g in zip(self._alpha, self._beta, self._gamma):
                self._compute_rhs(equations, time + a * self.dt)
                for eq in equations:
                    h = self._buffer("h", eq).mul_(b).add_(eq.rhs)
                    eq.variable.add_(h, alpha=g * self.dt)


@dataclass(frozen=True)
class TimeStepLimiter:
    target: float

    @abstractmethod
    def _compute(self, dt: float) -> float: ...

    def compute_max_time_step(self, dt: float) -> float: ...


@dataclass(frozen=True)
class CFLTimeStepLimiter(TimeStepLimiter):
    velocity: Field[Vector, Spectral]

    @override
    def _compute(self, dt: float) -> float:
        return (
            self.velocity.backend.reduction.global_max(fourier.to_physical(self.velocity))
            * dt
            / self.velocity.mesh.physical.spacing
        )


@dataclass(frozen=True)
class DiffusionTimeStepLimiter(TimeStepLimiter):
    diffusivity: float
    spacing: float

    @override
    def _compute(self, dt: float) -> float:
        return self.diffusivity * dt / self.spacing**2

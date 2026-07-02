from __future__ import annotations


from . import solver
from ..core import context


class SimulationControls:
    time: TimeControls

    def __init__(self, solver: solver.Solver):
        self.time = TimeControls(solver)


class TimeControls:
    current: float
    end: float
    save: SaveControls
    step: int
    solver: solver.Solver

    def __init__(self, solver: solver.Solver):
        self.solver = solver
        self.current = context.get().config.get('time/start', float)
        self.end = context.get().config.get('time/end', float)
        self.step = 0
        self.save = SaveControls()

    def advance(self) -> bool:
        return self.current < self.end

    def clamp_dt(self):
        if self.current + self.solver.integrator.dt > self.save.next:
            self.solver.integrator.overwrite_dt(self.save.next - self.current)
            self.save.activate()

    def update(self):
        self.current += self.solver.integrator.dt
        self.step += 1


class SaveControls:
    active: bool
    every: float
    next: float

    def __init__(self):
        self.active = False
        self.every = context.get().config.get('save/every', float)
        self.next = context.get().config.get('time/start', float) + self.every

    def activate(self) -> None:
        self.active = True
        self.next += self.every

    def deactivate(self) -> None:
        self.active = False

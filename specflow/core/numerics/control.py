from __future__ import annotations


from . import context
from .. import defs


class SimulationControls:
    time: TimeControls

    def __init__(self):
        self.time = TimeControls()


class TimeControls:
    current: float
    end: float
    save: SaveControls
    step: int

    def __init__(self):
        self.current = context.get().config.get('time/start', float)
        self.end = context.get().config.get('time/end', float)
        self.step = 0
        self.save = SaveControls(self.current)

    def advance(self) -> bool:
        return self.current < self.end

    def timestamp(self, time: float | None = None) -> str:
        if time is None:
            return f'{self.current:.{defs.TIMESTAMP_DIGITS}f}'
        else:
            return f'{time:.{defs.TIMESTAMP_DIGITS}f}'

    def update(self):
        _integ = context.get().sim.integrator
        if self.current + _integ.dt > self.save.next:
            _integ.overwrite_dt(self.save.next - self.current)
            self.save.activate()
        else:
            # To be moved to end of writing step
            self.save.deactivate()


class SaveControls:
    active: bool
    every: float
    next: float

    def __init__(self, current: float):
        self.active = True
        self.every = context.get().config.get('save/every', float)
        self.next = current + self.every

    def activate(self) -> None:
        self.active = True
        self.next += self.every

    def deactivate(self) -> None:
        self.active = False

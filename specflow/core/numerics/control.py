from __future__ import annotations

from specflow.simulation import integration


from . import get_context


class SimulationControls:
    time: TimeControls

    def __init__(self):
        self.time = TimeControls()


class TimeControls:
    current: float
    save: SaveControls
    step: int

    def __init__(self):
        self.current = get_context().config.get('time/start', float)
        self.step = 0

    def update(self, integrator: integration.TimeIntegrator):
        if self.current + integrator.dt > self.save.next:
            integrator.overwrite_dt(self.save.next - self.current)
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
        self.every = get_context().config.get('save/every', float)
        self.next = current + self.every

    def activate(self) -> None:
        self.active = True
        self.next += self.every

    def deactivate(self) -> None:
        self.active = False

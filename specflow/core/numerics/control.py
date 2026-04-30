from __future__ import annotations


from . import get_context


class SimulationControls:
    time: TimeControls

    def __init__(self):
        self.time = TimeControls()


class TimeControls:
    current: float
    step: int

    def __init__(self):
        self.current = get_context().config.get('time/start', float)
        self.step = 0

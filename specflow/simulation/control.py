from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationControls:
    time: TimeControls


@dataclass
class TimeControls:
    current: float
    end: float
    save: SaveControls
    step: int

    def advance(self) -> bool:
        return self.current < self.end

    def clamp_dt(self, target: float) -> float:
        if self.current + target > self.save.next:
            self.save.activate()
            return self.save.next - self.current
        else:
            return target

    def update(self, dt: float):
        self.current += dt
        self.step += 1


@dataclass
class SaveControls:
    every: float
    next: float
    active: bool = False

    def activate(self) -> None:
        self.active = True
        self.next += self.every

    def deactivate(self) -> None:
        self.active = False

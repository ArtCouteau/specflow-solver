from __future__ import annotations

from specflow.core.physics import dealias

from . import control, io, memory, mesh


class Context:
    config: io.UserConfiguration
    io: io.IOInterface
    memory: memory.MemoryManager
    mesh: mesh.Mesh
    physics: PhysicsContext
    sim: SimulationContext

    def __init__(self) -> None:
        self.memory = memory.MemoryManager()
        self.io = io.IOInterface()

    def read_user_configuration(self) -> None:
        self.config = io.UserConfiguration()

    def generate_mesh(self) -> None:
        self.mesh = mesh.Mesh()

    def generate_simulation_context(self) -> None:
        self.sim = SimulationContext()

    def generate_physics_context(self) -> None:
        self.physics = PhysicsContext(self)


class SimulationContext:
    control: control.SimulationControls

    def __init__(self) -> None:
        self.control = control.SimulationControls()


class PhysicsContext:
    dealias: dealias.Dealias

    def __init__(self, context: Context) -> None:
        self.dealias = dealias.DealiasRegistry.get(context.config.get('schemes/dealiasing', str))()


_context: Context | None = None


def init() -> None:
    print('Hello world')
    global _context
    _context = Context()
    _context.read_user_configuration()
    _context.generate_mesh()
    print('Done')


def get_context() -> Context:
    if _context is None:
        raise RuntimeError("Context has not been created; call specflow.init() first.")
    return _context

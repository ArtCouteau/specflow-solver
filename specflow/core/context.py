from __future__ import annotations

import logging

from .physics import dealias
from .numerics import io, log, memory, mesh
from .numerics.operations import derivative


class Context:
    config: io.UserConfiguration
    io: io.IOInterface
    log: logging.Logger
    memory: memory.MemoryManager
    mesh: mesh.Mesh
    physics: PhysicsContext

    def __init__(self) -> None:
        self.io = io.IOInterface()
        self.memory = memory.MemoryManager()

    def read_user_configuration(self) -> None:
        self.config = io.UserConfiguration()

    def configure_logger(self) -> None:
        self.log = log.configure()

    def generate_mesh(self) -> None:
        self.mesh = mesh.Mesh()

    def generate_physics_context(self) -> None:
        self.physics = PhysicsContext(self)


class PhysicsContext:
    dealias: dealias.Dealias
    derivative: derivative.DerivativeOperators

    def __init__(self, context: Context) -> None:
        self.dealias = dealias.DealiasRegistry.get(context.config.get('schemes/dealiasing', str))()
        self.derivative = derivative.DerivativeOperators()


_context: Context | None = None


def init() -> None:
    print('Initializing SPECFLOW solver')
    global _context
    _context = Context()
    _context.read_user_configuration()
    _context.generate_mesh()
    _context.generate_physics_context()
    print('Done')


def get() -> Context:
    if _context is None:
        raise RuntimeError("Context has not been created; call specflow.init() first.")
    return _context

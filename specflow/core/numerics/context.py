from __future__ import annotations

from . import io, memory, mesh


class Context:
    config: io.UserConfiguration
    io: io.IOInterface
    memory: memory.MemoryManager
    mesh: mesh.Mesh

    def __init__(self) -> None:
        self.memory = memory.MemoryManager()
        self.io = io.IOInterface()

    def read_user_configuration(self) -> None:
        self.config = io.UserConfiguration()


_context: Context | None = None


def init() -> None:
    print('Hello world')
    global _context
    _context = Context()
    _context.read_user_configuration()
    print('Done')


def generate_mesh() -> None:
    get_context().mesh = mesh.Mesh()


def get_context() -> Context:
    if _context is None:
        raise RuntimeError("Context has not been created; call specflow.init() first.")
    return _context

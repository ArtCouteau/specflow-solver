from __future__ import annotations

from . import io, memory


class Context:
    memory: memory.MemoryManager
    io: io.IOInterface
    config: io.UserConfiguration

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


def get_context() -> Context:
    if _context is None:
        raise RuntimeError("Context has not been created; call specflow.init() first.")
    return _context

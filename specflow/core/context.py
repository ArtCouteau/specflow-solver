from __future__ import annotations

from . import memory


class Context:
    def __init__(self) -> None:
        self.memory: memory.MemoryManager = memory.MemoryManager()


_context: Context | None = None


def init() -> None:
    print('Hello world')
    global _context
    _context = Context()
    print('Done')


def get_context() -> Context:
    if _context is None:
        raise RuntimeError("Context has not been created; call specflow.init() first.")
    return _context

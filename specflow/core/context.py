from __future__ import annotations

from specflow.core import memory


class Context():
    memory: memory.MemoryManager
    live: bool = False


def init():
    print('Hello world')
    Context.memory = memory.MemoryManager()
    Context.live = True
    print('Done')


def get_context() -> type[Context]:
    if not Context.live:
        raise RuntimeError("Context has not been created. Call specflow.init() first.")
    return Context

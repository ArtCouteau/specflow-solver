from __future__ import annotations

import os
import torch.distributed

from . import io, memory, mesh
from .operations import derivative
from ..physics import dealias


class Context:
    config: io.UserConfiguration
    io: io.IOInterface
    memory: memory.MemoryManager
    mesh: mesh.Mesh
    physics: PhysicsContext
    distribution: DistributionContext

    def __init__(self) -> None:
        self.io = io.IOInterface()
        self.memory = memory.MemoryManager()

    def read_user_configuration(self) -> None:
        self.config = io.UserConfiguration()

    def generate_distribution_context(self) -> None:
        self.distribution = DistributionContext(self)

    def generate_mesh(self) -> None:
        self.mesh = mesh.Mesh()

    def generate_physics_context(self) -> None:
        self.physics = PhysicsContext(self)


class DistributionContext:
    enabled: bool
    rank: int
    worldSize: int
    localRank: int

    def __init__(self, context: Context) -> None:
        self.enabled = context.config.get('numerics/distributed', bool, False)

        if self.enabled:
            torch.distributed.init_process_group(backend='nccl')
            self.rank = torch.distributed.get_rank()
            self.worldSize = torch.distributed.get_world_size()
            self.localRank = int(os.environ['LOCAL_RANK'])
        else:
            self.rank = 0
            self.worldSize = 1
            self.localRank = 0


class PhysicsContext:
    dealias: dealias.Dealias
    derivative: derivative.DerivativeOperators

    def __init__(self, context: Context) -> None:
        self.dealias = dealias.DealiasRegistry.get(context.config.get('schemes/dealiasing', str))()
        self.derivative = derivative.DerivativeOperators()


_context: Context | None = None


def init() -> None:
    print('Hello world')
    global _context
    _context = Context()
    _context.read_user_configuration()
    _context.generate_distribution_context()
    _context.generate_mesh()
    _context.generate_physics_context()
    print('Done')


def get() -> Context:
    if _context is None:
        raise RuntimeError("Context has not been created; call specflow.init() first.")
    return _context

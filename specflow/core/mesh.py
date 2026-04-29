from __future__ import annotations

import torch

from specflow.core import get_context


class Mesh():
    grid: torch.Tensor

    def __init__(self):
        self.grid = get_context().memory.alloc('mesh', (3, 8, 8), dtype=torch.int64)

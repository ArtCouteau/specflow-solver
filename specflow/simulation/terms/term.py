from __future__ import annotations

import torch


class Term:

    def __init__(self):
        pass

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor: ...

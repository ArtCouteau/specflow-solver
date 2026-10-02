from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class Quantity:
    data: torch.Tensor

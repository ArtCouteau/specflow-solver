from __future__ import annotations

import torch

from ...core.numerics import context
from ...core.numerics.operations import mask


class SpectralShells:
    kShell: torch.Tensor
    wavenumbers: torch.Tensor
    smoothing: torch.Tensor

    def __init__(self):
        kMax = context.get().config.get('grid/N', int)//2
        self.kShell = torch.arange(-0.5, kMax + 1.5).to(
            dtype=context.get().config.get('numerics/precision/float', torch.dtype),
            device=context.get().mesh.grid.device
        )
        self.kShell[0] = 0.0
        self.kShell[-1] = kMax
        self.wavenumbers = torch.arange(kMax + 1)
        self.smoothing = self._compute_smoothing_factor()

    @property
    def nShells(self) -> int:
        return self.kShell.size(0) - 1

    def _compute_smoothing_factor(self) -> torch.Tensor:
        smoothing = torch.zeros_like(
            self.wavenumbers,
            dtype=context.get().config.get('numerics/precision/float', torch.dtype)
        )

        totalN = mask.spherical(self.kShell[-1].item()).sum().item()
        totalV = (4.0/3.0)*torch.pi*(self.kShell[-1]**3)
        rhoNode = totalN/totalV

        for i in range(1, self.kShell.size(0) - 1):
            shell = self.shell(i)
            shellN = shell.sum().item()
            shellV = (4.0/3.0)*torch.pi*(self.kShell[i+1]**3 - self.kShell[i]**3)
            smoothing[i] = (shellV/shellN)*rhoNode

        return smoothing

    def spherical(self, index: int) -> torch.Tensor:
        return mask.spherical(self.kShell[index].item())

    def shell(self, index: int) -> torch.Tensor:
        return mask.shell(self.kShell[index].item(), self.kShell[index+1].item())

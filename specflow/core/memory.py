from __future__ import annotations

import torch


class MemoryManager:
    _buffers: dict[str, torch.Tensor]

    def __init__(self):
        self._buffers = {}

    def alloc(
            self,
            key: str,
            shape: tuple[int, ...],
            dtype: torch.dtype,
            device: str | torch.device = 'cuda'
    ) -> torch.Tensor:
        self._buffers[key] = torch.empty(shape, dtype=dtype, device=device)

        return self._buffers[key]

    def free(self, key: str):
        if not key in self._buffers:
            raise KeyError(f'Tensor "{key}" is not allocated')
        else:
            del self._buffers[key]

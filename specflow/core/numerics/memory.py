from __future__ import annotations

import math

import torch


def _mib(nBytes: int) -> str:
    return f"{nBytes / 1024**2:.1f} MiB"


class MemoryManager:
    def __init__(self):
        self._buffers: dict[str, torch.Tensor] = {}

    def alloc(
        self, key: str, shape: tuple[int, ...], dtype: torch.dtype, device: str | torch.device = "cuda"
    ) -> torch.Tensor:
        if key in self._buffers:
            raise KeyError(f'Tensor "{key}" is already allocated; call free() first')

        device = torch.device(device)
        if device.type == "cuda":
            required = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
            free, _ = torch.cuda.mem_get_info(device)
            if required > free:
                raise MemoryError(f"Cannot allocate {_mib(required)} on {device}: only {_mib(free)} free")

        self._buffers[key] = torch.empty(shape, dtype=dtype, device=device)

        return self._buffers[key]

    def free(self, key: str):
        if key not in self._buffers:
            raise KeyError(f'Tensor "{key}" is not allocated')

        del self._buffers[key]

    def free_cuda_cache(self):
        torch.cuda.empty_cache()

    def status(self):
        by_device: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
        for key, t in self._buffers.items():
            by_device.setdefault(t.device, []).append((key, t))

        print("=== MemoryManager Status ===")
        if not self._buffers:
            print("  (no registered buffers)")
            return

        for device, buffers in by_device.items():
            print(f"\nDevice: {device}")
            print("  Buffers:")
            for key, t in buffers:
                print(f"    {key:30s}  {t.dtype!s:20s}  shape={tuple(t.shape)}  {_mib(t.nbytes)}")

            if device.type == "cuda":
                print(f"  PyTorch reserved : {_mib(torch.cuda.memory_reserved(device))}")
                print(f"  PyTorch allocated: {_mib(torch.cuda.memory_allocated(device))}")
                free, total = torch.cuda.mem_get_info(device)
                print(f"  Driver free      : {_mib(free)} / {_mib(total)}")


import torch

from ...core import context

def time(range: tuple[float, float], folder: str, file: str, tag: str) -> torch.Tensor:
    io = context.get().io
    _timestamps = io.output.list_times(folder, range)
    if not _timestamps:
        raise ValueError(f"No timestamps found in the range {range} for folder {folder}.")

    for i, time in enumerate(_timestamps):
        if i == 0:
            _average = io.h5.read_tensor(f'{folder}/{time}/{file}', tag)
        else:
            _average += (io.h5.read_tensor(f'{folder}/{time}/{file}', tag) - _average) / (i + 1)

    return _average
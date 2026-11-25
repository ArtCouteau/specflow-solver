r"""Statistics over a window of a run's outputs, written to a time-average file.

The window $[t_0, t_1]$ takes the snapshots and the time-series rows with
$t_0 \le t \le t_1$; each is a sample of its source. The time-average file,
``time_average.h5`` by convention, holds ``mean/<name>`` and
``standard_error/<name>`` for every sampled diagnostic of the snapshots and every
diagnostic of the time series, the shells ``shells/wavenumbers`` and
``shells/counts`` when the snapshots hold them, and the attributes ``start``,
``end``, ``snapshot_samples``, ``time_series_samples``, ``block_length``,
``points``, ``length``, ``dimension``, ``dtype`` and ``commit`` (that of the code
writing the file). Integer diagnostics, such as counts, average in float64.

The mean is unweighted over all samples of a source. The standard error splits
them into contiguous blocks of ``block_length`` samples, drops a trailing partial
block, and is $s / \sqrt{B}$, with $s$ the standard deviation of the $B$ block
means, $B - 1$ in its denominator.
"""

import math
from pathlib import Path

import h5py
import torch

from specflow.outputs import (
    RESERVED_NAMES,
    TIME_SERIES_NAME,
    code_commit,
    snapshot_paths,
)

TIME_AVERAGE_NAME = "time_average.h5"
MESH_ATTRIBUTES = ("points", "length", "dimension", "dtype")


def window_statistic(
    directory: Path | str,
    path: Path | str,
    start: float,
    end: float,
    block_length: int,
) -> None:
    """Write the time-average file of a window of an output directory.

    Args:
        directory: The output directory of a run.
        path: The time-average file, never overwritten.
        start: The start of the window.
        end: The end of the window.
        block_length: The number of samples in a block.

    Raises:
        FileExistsError: The time-average file exists.
        FileNotFoundError: The directory holds no snapshot.
        ValueError: The block length is below 1, a snapshot of the window holds
            other sampled diagnostics or shapes than the first snapshot, or a
            source with diagnostics holds fewer than two blocks in the window;
            nothing is written.
    """
    directory, path = Path(directory), Path(path)
    if block_length < 1:
        raise ValueError(f"block length must be at least 1, got {block_length}")
    if path.exists():
        raise FileExistsError(f"{path} already exists; choose another path")
    paths = snapshot_paths(directory)
    if not paths:
        raise FileNotFoundError(f"{directory} holds no snapshot")
    with h5py.File(paths[0], "r") as first:
        mesh = {name: first.attrs[name] for name in MESH_ATTRIBUTES}
        layout = _layout(_datasets(first, "diagnostics"))
    snapshots, snapshot_count, shells = _snapshot_samples(paths, start, end, layout)
    series, series_count = _time_series_samples(
        directory / TIME_SERIES_NAME, start, end
    )
    sources = {"snapshot": snapshots, "time series": series}
    counts = {"snapshot": snapshot_count, "time series": series_count}
    held = {"snapshot": bool(layout), "time series": bool(series)}
    for source in sources:
        if held[source] and counts[source] // block_length < 2:
            raise ValueError(
                f"the window [{start}, {end}] holds {counts[source]} {source} "
                f"samples, fewer than two blocks of {block_length}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w-") as average:
        average.attrs.update(
            {
                "start": start,
                "end": end,
                "snapshot_samples": counts["snapshot"],
                "time_series_samples": counts["time series"],
                "block_length": block_length,
                **mesh,
                "commit": code_commit(),
            }
        )
        for samples in sources.values():
            for name, values in samples.items():
                mean, standard_error = _mean_and_standard_error(values, block_length)
                average.create_dataset(f"mean/{name}", data=mean.numpy())
                average.create_dataset(
                    f"standard_error/{name}", data=standard_error.numpy()
                )
        for name, values in shells.items():
            average.create_dataset(f"shells/{name}", data=values.numpy())


def _snapshot_samples(
    paths: list[Path],
    start: float,
    end: float,
    layout: dict[str, tuple[int, ...]],
) -> tuple[dict[str, torch.Tensor], int, dict[str, torch.Tensor]]:
    samples: dict[str, list[torch.Tensor]] = {}
    shells: dict[str, torch.Tensor] = {}
    count = 0
    for path in paths:
        with h5py.File(path, "r") as snapshot:
            if not start <= _number(snapshot.attrs, "time") <= end:
                continue
            count += 1
            diagnostics = _datasets(snapshot, "diagnostics")
            found = _layout(diagnostics)
            if found != layout:
                raise ValueError(
                    f"{path} holds the sampled diagnostics {found}, "
                    f"expected {layout} as in the first snapshot"
                )
            for name, values in diagnostics.items():
                samples.setdefault(name, []).append(values)
            if not shells:
                shells = _datasets(snapshot, "shells")
    stacked = {name: torch.stack(values) for name, values in samples.items()}
    return stacked, count, shells


def _time_series_samples(
    path: Path, start: float, end: float
) -> tuple[dict[str, torch.Tensor], int]:
    with h5py.File(path, "r") as series:
        columns = _datasets(series, "/")
    times = columns["time"]
    inside = (times >= start) & (times <= end)
    samples = {
        name: column[inside]
        for name, column in columns.items()
        if name not in RESERVED_NAMES
    }
    return samples, int(inside.sum().item())


def _datasets(file: h5py.File, group_name: str) -> dict[str, torch.Tensor]:
    group = file.get(group_name)
    if not isinstance(group, h5py.Group):
        return {}
    datasets: dict[str, torch.Tensor] = {}
    for name in group:
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{file.filename} entry {name!r} is not a dataset")
        datasets[str(name)] = torch.as_tensor(dataset[()])
    return datasets


def _layout(datasets: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    return {name: tuple(values.shape) for name, values in datasets.items()}


def _mean_and_standard_error(
    samples: torch.Tensor, block_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples.is_floating_point():
        samples = samples.to(torch.float64)
    blocks = samples.shape[0] // block_length
    used = samples[: blocks * block_length].unflatten(0, (blocks, block_length))
    block_means = used.mean(dim=1)
    standard_error = block_means.std(dim=0, correction=1).div_(math.sqrt(blocks))
    return samples.mean(dim=0), standard_error


def _number(attrs: h5py.AttributeManager, name: str) -> float:
    return float(torch.as_tensor(attrs[name]).item())

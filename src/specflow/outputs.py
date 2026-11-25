"""Snapshots and the diagnostics time series of a simulation, in one directory.

Snapshots are named by a zero-padded index, ``snapshot_000000.h5``, and hold the
spectral state under ``state/<key>`` with the metadata as attributes: ``time``,
``step``, ``time_step`` (the controller's Δt proposed from the snapshot's state),
``points``, ``length``, ``dimension``, ``dtype`` and ``commit`` (empty outside a
git checkout). The time series, ``time_series.h5``, holds one resizable dataset
each for ``time``, ``step`` and every diagnostic, with one row per evaluation.
Fields go through the decomposition's gather and scatter, only rank 0 writes, and
nothing is ever deleted or overwritten.
"""

import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import torch

from specflow.mesh import Mesh
from specflow.terms import State

type Diagnostic = Callable[[State, float], torch.Tensor]

SNAPSHOT_PREFIX = "snapshot_"
TIME_SERIES_NAME = "time_series.h5"
RESERVED_NAMES = ("time", "step")


@dataclass(frozen=True)
class Snapshot:
    """A state loaded from a snapshot, with its time, step and controller Δt."""

    state: dict[str, torch.Tensor]
    time: float
    step: int
    time_step: float


class Outputs:
    """The output directory of a simulation, its snapshot interval and diagnostics.

    Args:
        path: The output directory, created if missing.
        snapshot_interval: The time between output times.
        diagnostics: Functions of (state, time), by name, written to the time series.
        diagnostics_interval: The number of steps n between diagnostic evaluations;
            they are evaluated on the steps that are multiples of n.
        resume: Continue from the latest snapshot of the directory, instead of
            refusing a directory that holds snapshots.
    """

    def __init__(
        self,
        path: Path | str,
        snapshot_interval: float,
        diagnostics: Mapping[str, Diagnostic] | None = None,
        diagnostics_interval: int = 1,
        resume: bool = False,
    ) -> None:
        if not snapshot_interval > 0:
            raise ValueError(
                f"snapshot interval must be positive, got {snapshot_interval}"
            )
        if diagnostics_interval < 1:
            raise ValueError(
                f"diagnostics interval must be at least 1, got {diagnostics_interval}"
            )
        self.path = Path(path)
        self.snapshot_interval = snapshot_interval
        self.diagnostics = dict(diagnostics or {})
        self.diagnostics_interval = diagnostics_interval
        self.resume = resume
        for name in RESERVED_NAMES:
            if name in self.diagnostics:
                raise ValueError(f"diagnostic name {name!r} is reserved")

    def latest_snapshot(self, mesh: Mesh) -> Snapshot:
        """Return the latest snapshot of the directory, loaded onto the mesh."""
        return load_snapshot(_latest_snapshot_path(self.path), mesh)

    def open(self, mesh: Mesh, time: float) -> "OutputWriter":
        """Return the writer of a run starting at ``time``, checking the directory."""
        return OutputWriter(self, mesh, time)


class OutputWriter:
    """The outputs of one run: snapshot numbering and buffered diagnostics.

    ``step`` is the step of the run's start and ``resumed`` whether the run
    continues from the latest snapshot, already written at its start time.
    """

    def __init__(self, outputs: Outputs, mesh: Mesh, time: float) -> None:
        self._outputs = outputs
        self._mesh = mesh
        self._writes = mesh.decomposition.rank == 0
        self._commit = _commit()
        self._time_series = outputs.path / TIME_SERIES_NAME
        self._times: list[float] = []
        self._steps: list[int] = []
        self._buffers: dict[str, list[torch.Tensor]] = {
            name: [] for name in outputs.diagnostics
        }
        self.resumed = outputs.resume
        if self.resumed:
            self.step, self._index = self._check_resume(time)
        else:
            self.step, self._index = 0, 0
            self._check_fresh()

    def sample(self, state: State, time: float, step: int) -> None:
        """Buffer copies of the diagnostics on device if ``step`` is a multiple of n."""
        if step % self._outputs.diagnostics_interval:
            return
        self._times.append(time)
        self._steps.append(step)
        for name, diagnostic in self._outputs.diagnostics.items():
            self._buffers[name].append(diagnostic(state, time).clone())

    def write(self, state: State, time: float, step: int, time_step: float) -> None:
        """Write the buffered diagnostics, then the snapshot of an output time.

        Raises:
            FloatingPointError: A buffered diagnostic is not finite.
        """
        self._flush()
        mesh = self._mesh
        gathered = {key: mesh.decomposition.gather(t) for key, t in state.items()}
        if self._writes:
            path = _snapshot_path(self._outputs.path, self._index)
            with h5py.File(path, "w-") as snapshot:
                snapshot.attrs.update(
                    {
                        "time": time,
                        "step": step,
                        "time_step": time_step,
                        "points": mesh.points,
                        "length": mesh.length,
                        "dimension": mesh.dimension,
                        "dtype": _dtype_name(mesh),
                        "commit": self._commit,
                    }
                )
                for key, tensor in gathered.items():
                    snapshot.create_dataset(f"state/{key}", data=tensor.cpu().numpy())
        self._index += 1

    def _check_fresh(self) -> None:
        path = self._outputs.path
        if _snapshot_indices(path):
            raise FileExistsError(
                f"{path} already holds snapshots; resume or choose another directory"
            )
        if self._writes:
            path.mkdir(parents=True, exist_ok=True)
            with h5py.File(self._time_series, "w-"):
                pass

    def _check_resume(self, time: float) -> tuple[int, int]:
        latest = _latest_snapshot_path(self._outputs.path)
        with h5py.File(latest, "r") as snapshot:
            latest_time = _number(snapshot.attrs, "time")
            latest_step = int(_number(snapshot.attrs, "step"))
        if time != latest_time:
            raise ValueError(
                f"a resumed run must start at the latest snapshot's time "
                f"{latest_time}, got {time}"
            )
        with h5py.File(self._time_series, "r") as series:
            stored = set(series) - set(RESERVED_NAMES)
        if stored != set(self._outputs.diagnostics):
            raise ValueError(
                f"the time series holds the diagnostics {sorted(stored)}, "
                f"expected {sorted(self._outputs.diagnostics)}"
            )
        return latest_step, _snapshot_indices(self._outputs.path)[-1] + 1

    def _flush(self) -> None:
        if not self._steps:
            return
        columns = {
            name: torch.stack(buffer).cpu() for name, buffer in self._buffers.items()
        }
        self._check_finite(columns)
        if self._writes:
            with h5py.File(self._time_series, "r+") as series:
                _append(series, "time", torch.tensor(self._times, dtype=torch.float64))
                _append(series, "step", torch.tensor(self._steps, dtype=torch.int64))
                for name, column in columns.items():
                    _append(series, name, column)
        self._times.clear()
        self._steps.clear()
        for buffer in self._buffers.values():
            buffer.clear()

    def _check_finite(self, columns: Mapping[str, torch.Tensor]) -> None:
        for row in range(len(self._steps)):
            for name, column in columns.items():
                if not torch.isfinite(column[row]).all():
                    raise FloatingPointError(
                        f"diagnostic {name!r} is not finite at step "
                        f"{self._steps[row]}, time {self._times[row]:.6g}"
                    )


def load_snapshot(path: Path, mesh: Mesh) -> Snapshot:
    """Return a snapshot, its state scattered onto the mesh's device.

    Raises:
        ValueError: The snapshot was written on a mesh of other N, L, dimension or
            dtype.
    """
    with h5py.File(path, "r") as snapshot:
        attrs = snapshot.attrs
        expected: dict[str, float | str] = {
            "points": mesh.points,
            "length": mesh.length,
            "dimension": mesh.dimension,
        }
        stored: dict[str, float | str] = {
            name: _number(attrs, name) for name in expected
        }
        expected["dtype"] = _dtype_name(mesh)
        stored["dtype"] = str(attrs["dtype"])
        for name, value in expected.items():
            if stored[name] != value:
                raise ValueError(
                    f"snapshot {path} has {name} {stored[name]}, expected {value}"
                )
        group = snapshot["state"]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"snapshot {path} holds no state group")
        state = {str(key): _load_field(group, str(key), mesh) for key in group}
        return Snapshot(
            state,
            _number(attrs, "time"),
            int(_number(attrs, "step")),
            _number(attrs, "time_step"),
        )


def _load_field(group: h5py.Group, key: str, mesh: Mesh) -> torch.Tensor:
    dataset = group[key]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"state key {key!r} of a snapshot is not a dataset")
    full_spectral = torch.from_numpy(dataset[()]).to(mesh.wavenumbers.device)
    return mesh.decomposition.scatter(full_spectral)


def _append(series: h5py.File, name: str, column: torch.Tensor) -> None:
    if name not in series:
        series.create_dataset(
            name, data=column.numpy(), maxshape=(None, *column.shape[1:]), chunks=True
        )
        return
    dataset = series[name]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"time series entry {name!r} is not a dataset")
    rows = dataset.shape[0]
    dataset.resize(rows + column.shape[0], axis=0)
    dataset[rows:] = column.numpy()


def _number(attrs: h5py.AttributeManager, name: str) -> float:
    return float(torch.as_tensor(attrs[name]).item())


def _dtype_name(mesh: Mesh) -> str:
    return str(mesh.dtype).removeprefix("torch.")


def _snapshot_path(directory: Path, index: int) -> Path:
    return directory / f"{SNAPSHOT_PREFIX}{index:06d}.h5"


def _latest_snapshot_path(directory: Path) -> Path:
    indices = _snapshot_indices(directory)
    if not indices:
        raise FileNotFoundError(f"{directory} holds no snapshot to resume from")
    return _snapshot_path(directory, indices[-1])


def _snapshot_indices(directory: Path) -> list[int]:
    return sorted(
        int(path.stem.removeprefix(SNAPSHOT_PREFIX))
        for path in directory.glob(f"{SNAPSHOT_PREFIX}*.h5")
    )


def _commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""

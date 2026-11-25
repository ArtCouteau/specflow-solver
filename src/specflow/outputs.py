"""Snapshots and the diagnostics time series of a simulation, in one directory.

Snapshots are named by a zero-padded index, ``snapshot_000000.h5``, and hold the
spectral state under ``state/<key>`` with the metadata as attributes. Every field
goes through the decomposition's gather and scatter, and only rank 0 writes.
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

SNAPSHOT_GLOB = "snapshot_*.h5"
TIME_SERIES_NAME = "time_series.h5"
RESERVED_NAMES = ("time", "step")


@dataclass(frozen=True)
class Snapshot:
    """A state loaded from a snapshot, with the time, step and controller Δt."""

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
        diagnostics_interval: The number of steps between diagnostic evaluations.
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
        for name in RESERVED_NAMES:
            if diagnostics is not None and name in diagnostics:
                raise ValueError(f"diagnostic name {name!r} is reserved")
        self.path = Path(path)
        self.snapshot_interval = snapshot_interval
        self.diagnostics = dict(diagnostics or {})
        self.diagnostics_interval = diagnostics_interval
        self.resume = resume

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
        self._commit = _commit()
        self.step = 0
        self.resumed = False
        self._index = 0
        self._times: list[float] = []
        self._steps: list[int] = []
        self._buffers: dict[str, list[torch.Tensor]] = {
            name: [] for name in outputs.diagnostics
        }
        self._time_series = outputs.path / TIME_SERIES_NAME
        if mesh.decomposition.rank == 0:
            outputs.path.mkdir(parents=True, exist_ok=True)
            if _snapshot_indices(outputs.path):
                raise FileExistsError(
                    f"{outputs.path} already holds snapshots; resume or choose "
                    "another directory"
                )
            with h5py.File(self._time_series, "w-"):
                pass

    def sample(self, state: State, time: float, step: int) -> None:
        """Evaluate the diagnostics on every n-th step, buffered on the device."""
        if step % self._outputs.diagnostics_interval:
            return
        self._times.append(time)
        self._steps.append(step)
        for name, diagnostic in self._outputs.diagnostics.items():
            self._buffers[name].append(diagnostic(state, time))

    def write(self, state: State, time: float, step: int, time_step: float) -> None:
        """Write the buffered diagnostics, then the snapshot of an output time.

        Raises:
            FloatingPointError: A buffered diagnostic is not finite.
        """
        self._flush()
        mesh = self._mesh
        gathered = {key: mesh.decomposition.gather(t) for key, t in state.items()}
        if mesh.decomposition.rank == 0:
            path = self._outputs.path / f"snapshot_{self._index:06d}.h5"
            with h5py.File(path, "w-") as snapshot:
                snapshot.attrs.update(
                    {
                        "time": time,
                        "step": step,
                        "time_step": time_step,
                        "points": mesh.points,
                        "length": mesh.length,
                        "dimension": mesh.dimension,
                        "dtype": str(mesh.dtype).removeprefix("torch."),
                        "commit": self._commit,
                    }
                )
                for key, tensor in gathered.items():
                    snapshot.create_dataset(f"state/{key}", data=tensor.cpu().numpy())
        self._index += 1

    def _flush(self) -> None:
        if not self._steps:
            return
        columns = {
            name: torch.stack(buffer).cpu() for name, buffer in self._buffers.items()
        }
        self._check_finite(columns)
        if self._mesh.decomposition.rank == 0:
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


def _append(series: h5py.File, name: str, column: torch.Tensor) -> None:
    if name not in series:
        series.create_dataset(
            name, data=column.numpy(), maxshape=(None, *column.shape[1:]), chunks=True
        )
        return
    dataset = series[name]
    rows = dataset.shape[0]
    dataset.resize(rows + column.shape[0], axis=0)
    dataset[rows:] = column.numpy()


def _snapshot_indices(directory: Path) -> list[int]:
    return sorted(
        int(path.stem.removeprefix("snapshot_"))
        for path in directory.glob(SNAPSHOT_GLOB)
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

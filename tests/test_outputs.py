import logging
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import pytest
import torch

import specflow
from specflow import diagnostics, flows
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4
from specflow.mesh import Mesh
from specflow.outputs import Outputs, load_snapshot
from specflow.simulation import Simulation
from specflow.terms import Diffusion, State
from specflow.time_step import FixedTimeStep

POINTS = 16
LENGTH = 2 * math.pi
VISCOSITY = 0.5
AMPLITUDES = {1: 1.0, 2: -0.5, 3: 0.25}
TIME_STEP = 0.1
INTERVAL = 0.25


def make_mesh() -> Mesh:
    return Mesh(POINTS, LENGTH, 1, SerialDecomposition(POINTS, 1))


def exact_spectral(mesh: Mesh, time: float) -> torch.Tensor:
    solution_physical = flows.diffusion_1d_physical(mesh, AMPLITUDES, VISCOSITY, time)
    return mesh.decomposition.to_spectral(solution_physical.unsqueeze(0))


def exact_energy(time: float) -> float:
    return sum(
        0.25 * amplitude**2 * math.exp(-2 * VISCOSITY * integer**2 * time)
        for integer, amplitude in AMPLITUDES.items()
    )


def exact_dissipation(time: float) -> float:
    return sum(
        0.5
        * VISCOSITY
        * (integer * amplitude) ** 2
        * math.exp(-2 * VISCOSITY * integer**2 * time)
        for integer, amplitude in AMPLITUDES.items()
    )


def energy_outputs(
    mesh: Mesh, path: Path, diagnostics_interval: int = 1, resume: bool = False
) -> Outputs:
    def energy(state: State, time: float) -> torch.Tensor:
        return diagnostics.kinetic_energy(mesh, state["velocity"])

    def dissipation(state: State, time: float) -> torch.Tensor:
        return diagnostics.dissipation(mesh, state["velocity"], VISCOSITY)

    named = {"energy": energy, "dissipation": dissipation}
    return Outputs(path, INTERVAL, named, diagnostics_interval, resume)


def make_simulation(mesh: Mesh, outputs: Outputs) -> Simulation:
    equation = Equation(
        "velocity", [Diffusion(mesh, "velocity", VISCOSITY, components=(1,))]
    )
    return Simulation(mesh, [equation], IFRK4(), FixedTimeStep(TIME_STEP), outputs)


def read_dataset(file: h5py.File, name: str) -> torch.Tensor:
    dataset = file[name]
    assert isinstance(dataset, h5py.Dataset)
    return torch.from_numpy(dataset[()])


def read_time_series(directory: Path) -> dict[str, torch.Tensor]:
    with h5py.File(directory / "time_series.h5", "r") as series:
        return {str(name): read_dataset(series, str(name)) for name in series}


def snapshot_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("snapshot_*.h5"))


def test_snapshots_sit_at_exact_output_times_with_their_metadata(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    outputs = Outputs(tmp_path, INTERVAL)

    make_simulation(mesh, outputs).run({"velocity": exact_spectral(mesh, 0.0)}, 0, 1)

    paths = snapshot_paths(tmp_path)
    assert [path.name for path in paths] == [
        f"snapshot_{index:06d}.h5" for index in range(5)
    ]
    for index, path in enumerate(paths):
        with h5py.File(path, "r") as snapshot:
            assert snapshot.attrs["time"] == index * INTERVAL
            assert snapshot.attrs["step"] == 3 * index
            assert snapshot.attrs["time_step"] == TIME_STEP
            assert snapshot.attrs["points"] == POINTS
            assert snapshot.attrs["length"] == LENGTH
            assert snapshot.attrs["dimension"] == 1
            assert snapshot.attrs["dtype"] == "float64"
            assert isinstance(snapshot.attrs["commit"], str)
            stored = read_dataset(snapshot, "state/velocity")
        expected = exact_spectral(mesh, index * INTERVAL)
        assert (stored - expected).abs().max().item() <= 1e-14


def test_a_run_ending_between_output_times_writes_a_snapshot_at_its_end(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    outputs = Outputs(tmp_path, INTERVAL)

    make_simulation(mesh, outputs).run({"velocity": exact_spectral(mesh, 0)}, 0, 0.6)

    times = [load_snapshot(path, mesh).time for path in snapshot_paths(tmp_path)]
    assert times == [0.0, 0.25, 0.5, 0.6]


def test_the_time_series_holds_the_diagnostics_every_n_steps(tmp_path: Path) -> None:
    mesh = make_mesh()
    outputs = energy_outputs(mesh, tmp_path, diagnostics_interval=2)

    make_simulation(mesh, outputs).run({"velocity": exact_spectral(mesh, 0)}, 0, 1)

    assert sorted(
        path.name for path in tmp_path.glob("*.h5") if "snapshot" not in path.name
    ) == ["time_series.h5"]
    with h5py.File(tmp_path / "time_series.h5", "r") as series:
        steps = read_dataset(series, "step").tolist()
        times = read_dataset(series, "time").tolist()
        energy = read_dataset(series, "energy").tolist()
        dissipation = read_dataset(series, "dissipation").tolist()
    assert steps == list(range(0, 13, 2))
    assert times == pytest.approx([0, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0], abs=1e-14)
    assert energy == pytest.approx([exact_energy(t) for t in times], rel=1e-12)
    assert dissipation == pytest.approx(
        [exact_dissipation(t) for t in times], rel=1e-12
    )


def test_a_diagnostic_returning_a_view_of_the_state_keeps_its_value(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()

    def first_mode(state: State, time: float) -> torch.Tensor:
        return state["velocity"][..., 1].imag

    outputs = Outputs(tmp_path, INTERVAL, {"first mode": first_mode})

    make_simulation(mesh, outputs).run({"velocity": exact_spectral(mesh, 0)}, 0, 1)

    series = read_time_series(tmp_path)
    expected = [exact_spectral(mesh, t)[..., 1].imag for t in series["time"].tolist()]
    assert (series["first mode"] - torch.stack(expected)).abs().max().item() <= 1e-14


def test_a_snapshot_loads_as_the_state_of_a_restart(tmp_path: Path) -> None:
    mesh = make_mesh()
    simulation = make_simulation(mesh, Outputs(tmp_path / "first", INTERVAL))
    simulation.run({"velocity": exact_spectral(mesh, 0)}, 0, 0.5)

    snapshot = load_snapshot(tmp_path / "first" / "snapshot_000001.h5", mesh)
    restart = make_simulation(mesh, Outputs(tmp_path / "second", INTERVAL))
    final = restart.run(snapshot.state, snapshot.time, 1.0)

    assert (snapshot.time, snapshot.step, snapshot.time_step) == (0.25, 3, TIME_STEP)
    expected = exact_spectral(mesh, 1.0)
    assert (final["velocity"] - expected).abs().max().item() <= 1e-14


def test_a_snapshot_of_another_mesh_raises(tmp_path: Path) -> None:
    mesh = make_mesh()
    make_simulation(mesh, Outputs(tmp_path, INTERVAL)).run(
        {"velocity": exact_spectral(mesh, 0)}, 0, 0
    )
    other = Mesh(POINTS, 1.0, 1, SerialDecomposition(POINTS, 1))

    with pytest.raises(ValueError, match="length"):
        load_snapshot(tmp_path / "snapshot_000000.h5", other)


def test_a_fresh_run_refuses_a_directory_holding_snapshots(tmp_path: Path) -> None:
    mesh = make_mesh()
    initial = {"velocity": exact_spectral(mesh, 0)}
    make_simulation(mesh, Outputs(tmp_path, INTERVAL)).run(initial, 0, 0.5)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    with pytest.raises(FileExistsError, match="snapshots"):
        make_simulation(mesh, Outputs(tmp_path, INTERVAL)).run(initial, 0, 0.5)

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_files_named_like_snapshots_without_an_index_are_ignored(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    for name in ("snapshot_backup.h5", "snapshot_.h5", "snapshot_1a.h5"):
        (tmp_path / name).write_bytes(b"not a snapshot")
    outputs = Outputs(tmp_path, INTERVAL)

    make_simulation(mesh, outputs).run({"velocity": exact_spectral(mesh, 0)}, 0, 1)

    assert outputs.latest_snapshot(mesh).time == 1
    assert (tmp_path / "snapshot_backup.h5").read_bytes() == b"not a snapshot"


def test_a_fresh_run_never_overwrites_a_time_series(tmp_path: Path) -> None:
    mesh = make_mesh()
    (tmp_path / "time_series.h5").write_bytes(b"precious")

    with pytest.raises(FileExistsError):
        make_simulation(mesh, Outputs(tmp_path, INTERVAL)).run(
            {"velocity": exact_spectral(mesh, 0)}, 0, 1
        )

    assert (tmp_path / "time_series.h5").read_bytes() == b"precious"


def test_resume_continues_numbering_and_time_series(tmp_path: Path) -> None:
    mesh = make_mesh()
    initial = {"velocity": exact_spectral(mesh, 0)}
    make_simulation(mesh, energy_outputs(mesh, tmp_path / "whole")).run(initial, 0, 1)
    make_simulation(mesh, energy_outputs(mesh, tmp_path / "split")).run(initial, 0, 0.5)

    outputs = energy_outputs(mesh, tmp_path / "split", resume=True)
    latest = outputs.latest_snapshot(mesh)
    final = make_simulation(mesh, outputs).run(latest.state, latest.time, 1)

    whole = [load_snapshot(p, mesh) for p in snapshot_paths(tmp_path / "whole")]
    split = [load_snapshot(p, mesh) for p in snapshot_paths(tmp_path / "split")]
    assert [(s.time, s.step) for s in split] == [(s.time, s.step) for s in whole]
    assert torch.equal(final["velocity"], whole[-1].state["velocity"])
    whole_series = read_time_series(tmp_path / "whole")
    split_series = read_time_series(tmp_path / "split")
    assert whole_series.keys() == split_series.keys()
    for name, column in whole_series.items():
        assert torch.equal(split_series[name], column)


@dataclass(frozen=True)
class FailingGatherDecomposition(SerialDecomposition):
    failing_call: int
    calls: list[int] = field(default_factory=list[int])

    def gather(self, field_spectral: torch.Tensor) -> torch.Tensor:
        self.calls.append(len(self.calls))
        if len(self.calls) == self.failing_call:
            raise OSError("gather failed")
        return field_spectral


def test_resume_after_a_failed_output_time_continues_the_time_series(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    initial = {"velocity": exact_spectral(mesh, 0)}
    make_simulation(mesh, energy_outputs(mesh, tmp_path / "whole")).run(initial, 0, 1)
    failing = Mesh(POINTS, LENGTH, 1, FailingGatherDecomposition(POINTS, 1, 3))
    with pytest.raises(OSError, match="gather failed"):
        make_simulation(failing, energy_outputs(failing, tmp_path / "split")).run(
            initial, 0, 1
        )

    outputs = energy_outputs(mesh, tmp_path / "split", resume=True)
    latest = outputs.latest_snapshot(mesh)
    make_simulation(mesh, outputs).run(latest.state, latest.time, 1)

    whole_series = read_time_series(tmp_path / "whole")
    split_series = read_time_series(tmp_path / "split")
    for name, column in whole_series.items():
        assert torch.equal(split_series[name], column)


def test_resume_with_time_series_rows_past_the_latest_snapshot_raises(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    make_simulation(mesh, energy_outputs(mesh, tmp_path)).run(
        {"velocity": exact_spectral(mesh, 0)}, 0, 0.5
    )
    with h5py.File(tmp_path / "time_series.h5", "r+") as series:
        for name in series:
            dataset = series[name]
            assert isinstance(dataset, h5py.Dataset)
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = 7 if name == "step" else 0.0
    outputs = energy_outputs(mesh, tmp_path, resume=True)
    latest = outputs.latest_snapshot(mesh)

    with pytest.raises(ValueError, match="step 7"):
        make_simulation(mesh, outputs).run(latest.state, latest.time, 1)


def test_a_run_failing_before_its_first_snapshot_leaves_the_directory_fresh(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    initial = {"velocity": exact_spectral(mesh, 0)}

    def not_a_number(state: State, time: float) -> torch.Tensor:
        return torch.tensor(math.nan)

    with pytest.raises(FloatingPointError):
        make_simulation(mesh, Outputs(tmp_path, INTERVAL, {"nan": not_a_number})).run(
            initial, 0, 1
        )
    assert list(tmp_path.iterdir()) == []

    make_simulation(mesh, energy_outputs(mesh, tmp_path)).run(initial, 0, 1)

    assert len(snapshot_paths(tmp_path)) == 5


def test_resume_from_a_directory_without_snapshots_raises(tmp_path: Path) -> None:
    mesh = make_mesh()
    outputs = Outputs(tmp_path, INTERVAL, resume=True)

    with pytest.raises(FileNotFoundError, match="no snapshot"):
        make_simulation(mesh, outputs).run({"velocity": exact_spectral(mesh, 0)}, 0, 1)


def test_resume_from_another_time_than_the_latest_snapshot_raises(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    initial = {"velocity": exact_spectral(mesh, 0)}
    make_simulation(mesh, Outputs(tmp_path, INTERVAL)).run(initial, 0, 0.5)
    outputs = Outputs(tmp_path, INTERVAL, resume=True)

    with pytest.raises(ValueError, match="latest snapshot"):
        make_simulation(mesh, outputs).run(initial, 0, 1)


def test_resume_with_other_diagnostics_raises(tmp_path: Path) -> None:
    mesh = make_mesh()
    make_simulation(mesh, energy_outputs(mesh, tmp_path)).run(
        {"velocity": exact_spectral(mesh, 0)}, 0, 0.5
    )
    outputs = Outputs(tmp_path, INTERVAL, resume=True)
    latest = outputs.latest_snapshot(mesh)

    with pytest.raises(ValueError, match="diagnostics"):
        make_simulation(mesh, outputs).run(latest.state, latest.time, 1)


def test_a_non_finite_diagnostic_raises_with_its_step_and_time(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()

    def blowing_up(state: State, time: float) -> torch.Tensor:
        return torch.tensor(math.nan if time > 0.3 else 1.0)

    outputs = Outputs(tmp_path, INTERVAL, {"blowing up": blowing_up})

    with pytest.raises(FloatingPointError, match="'blowing up'.*step 4, time 0.35"):
        make_simulation(mesh, outputs).run({"velocity": exact_spectral(mesh, 0)}, 0, 1)


def test_one_info_line_per_output_time(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mesh = make_mesh()
    simulation = make_simulation(mesh, Outputs(tmp_path, INTERVAL))

    with caplog.at_level(logging.DEBUG, logger="specflow"):
        simulation.run({"velocity": exact_spectral(mesh, 0)}, 0, 1)

    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert [r.name for r in info] == 5 * ["specflow.simulation"]
    assert [r.getMessage().split(",")[0] for r in info] == [
        f"step {step}" for step in range(0, 13, 3)
    ]


def test_each_debug_line_holds_the_time_step_taken(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mesh = make_mesh()
    simulation = make_simulation(mesh, Outputs(tmp_path, INTERVAL))

    with caplog.at_level(logging.DEBUG, logger="specflow"):
        simulation.run({"velocity": exact_spectral(mesh, 0)}, 0, 0.5)

    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert [message.split("time_step ")[1] for message in debug] == 2 * [
        "0.1",
        "0.1",
        "0.05",
    ]


def test_the_library_never_configures_logging(tmp_path: Path) -> None:
    mesh = make_mesh()

    make_simulation(mesh, Outputs(tmp_path, INTERVAL)).run(
        {"velocity": exact_spectral(mesh, 0)}, 0, 1
    )

    for name in ("specflow", "specflow.simulation", "specflow.outputs"):
        logger = logging.getLogger(name)
        assert (logger.handlers, logger.level) == ([], logging.NOTSET)


def commit_of(path: Path, mesh: Mesh) -> str:
    with h5py.File(path, "r") as snapshot:
        commit = snapshot.attrs["commit"]
    assert isinstance(commit, str)
    return commit


def test_the_code_commit_is_stored_in_a_git_checkout(tmp_path: Path) -> None:
    mesh = make_mesh()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    make_simulation(mesh, Outputs(tmp_path, INTERVAL)).run(
        {"velocity": exact_spectral(mesh, 0)}, 0, 0
    )

    assert commit_of(tmp_path / "snapshot_000000.h5", mesh) == head


def test_the_code_commit_is_empty_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mesh = make_mesh()
    monkeypatch.setenv("PATH", "")

    make_simulation(mesh, Outputs(tmp_path, INTERVAL)).run(
        {"velocity": exact_spectral(mesh, 0)}, 0, 0
    )

    assert commit_of(tmp_path / "snapshot_000000.h5", mesh) == ""


OUTER_RUN = """
import sys
from pathlib import Path

import specflow
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4
from specflow.mesh import Mesh
from specflow.outputs import Outputs
from specflow.simulation import Simulation
from specflow.terms import Diffusion
from specflow.time_step import FixedTimeStep

assert Path(specflow.__file__).is_relative_to(sys.argv[2])
mesh = Mesh(8, 1.0, 1, SerialDecomposition(8, 1))
equation = Equation("scalar", [Diffusion(mesh, "scalar", 0.1)])
simulation = Simulation(
    mesh, [equation], IFRK4(), FixedTimeStep(0.1), Outputs(sys.argv[1], 1.0)
)
simulation.run({"scalar": mesh.decomposition.to_spectral(mesh.coordinates[0])}, 0, 0)
"""


def test_the_code_commit_is_empty_inside_another_repository(tmp_path: Path) -> None:
    mesh = make_mesh()
    outer = tmp_path / "outer"
    packages = outer / "site-packages"
    shutil.copytree(Path(specflow.__file__).parent, packages / "specflow")
    git = ["git", "-c", "user.name=outer", "-c", "user.email=outer@example.com"]
    git += ["-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], cwd=outer, check=True)
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "-m", "outer"], cwd=outer, check=True
    )

    subprocess.run(
        [sys.executable, "-c", OUTER_RUN, tmp_path, packages],
        env={**os.environ, "PYTHONPATH": str(packages)},
        check=True,
    )

    assert commit_of(tmp_path / "snapshot_000000.h5", mesh) == ""


@dataclass(frozen=True)
class OtherRankDecomposition(SerialDecomposition):
    @property
    def rank(self) -> int:
        return 1


def test_only_rank_0_writes(tmp_path: Path) -> None:
    mesh = Mesh(POINTS, LENGTH, 1, OtherRankDecomposition(POINTS, 1))

    make_simulation(mesh, Outputs(tmp_path / "outputs", INTERVAL)).run(
        {"velocity": exact_spectral(mesh, 0)}, 0, 1
    )

    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("interval", "diagnostics_interval", "name", "match"),
    [
        (0.0, 1, "energy", "snapshot interval"),
        (math.inf, 1, "energy", "snapshot interval"),
        (math.nan, 1, "energy", "snapshot interval"),
        (INTERVAL, 0, "energy", "diagnostics interval"),
        (INTERVAL, 1, "time", "reserved"),
        (INTERVAL, 1, "step", "reserved"),
    ],
)
def test_invalid_outputs_raise(
    tmp_path: Path, interval: float, diagnostics_interval: int, name: str, match: str
) -> None:
    def zero(state: State, time: float) -> torch.Tensor:
        return torch.zeros(())

    with pytest.raises(ValueError, match=match):
        Outputs(tmp_path, interval, {name: zero}, diagnostics_interval)

import math
from pathlib import Path

import h5py
import pytest
import torch

from specflow import diagnostics, flows
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4
from specflow.mesh import Mesh
from specflow.outputs import Outputs
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


def energy_outputs(mesh: Mesh, path: Path, **options: bool | int) -> Outputs:
    def energy(state: State, time: float) -> torch.Tensor:
        return diagnostics.kinetic_energy(mesh, state["velocity"])

    def dissipation(state: State, time: float) -> torch.Tensor:
        return diagnostics.dissipation(mesh, state["velocity"], VISCOSITY)

    named = {"energy": energy, "dissipation": dissipation}
    return Outputs(path, INTERVAL, named, **options)


def make_simulation(mesh: Mesh, outputs: Outputs) -> Simulation:
    equation = Equation(
        "velocity", [Diffusion(mesh, "velocity", VISCOSITY, components=(1,))]
    )
    return Simulation(mesh, [equation], IFRK4(), FixedTimeStep(TIME_STEP), outputs)


def read_dataset(file: h5py.File, name: str) -> torch.Tensor:
    return torch.from_numpy(file[name][()])


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

    times: list[float] = []
    for path in snapshot_paths(tmp_path):
        with h5py.File(path, "r") as snapshot:
            times.append(float(snapshot.attrs["time"]))
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

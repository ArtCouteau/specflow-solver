import math
import subprocess
from pathlib import Path

import h5py
import pytest
import torch

from specflow import diagnostics, flows, spectra
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4
from specflow.mesh import Mesh
from specflow.outputs import Outputs
from specflow.simulation import Simulation
from specflow.spectra import Shells
from specflow.statistics import window_statistic
from specflow.terms import State
from specflow.terms import Diffusion
from specflow.time_step import FixedTimeStep

POINTS = 16
LENGTH = 2 * math.pi
VISCOSITY = 0.5
AMPLITUDES = {1: 1.0, 2: -0.5, 3: 0.25}
TIME_STEP = 0.1
INTERVAL = 0.25
COMPONENTS = 4096


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


def exact_spectrum(time: float) -> torch.Tensor:
    spectrum = torch.zeros(POINTS // 2 + 1, dtype=torch.float64)
    for integer, amplitude in AMPLITUDES.items():
        spectrum[integer] = (
            0.25 * amplitude**2 * math.exp(-2 * VISCOSITY * integer**2 * time)
        )
    return spectrum


def run(mesh: Mesh, outputs: Outputs, end: float) -> None:
    equation = Equation(
        "velocity", [Diffusion(mesh, "velocity", VISCOSITY, components=(1,))]
    )
    simulation = Simulation(
        mesh, [equation], IFRK4(), FixedTimeStep(TIME_STEP), outputs
    )
    simulation.run({"velocity": exact_spectral(mesh, 0)}, 0, end)


def spectrum_outputs(mesh: Mesh, path: Path) -> Outputs:
    shells = Shells(mesh)

    def energy(state: State, time: float) -> torch.Tensor:
        return diagnostics.kinetic_energy(mesh, state["velocity"])

    def spectrum(state: State, time: float) -> torch.Tensor:
        return spectra.energy_spectrum(shells, state["velocity"])

    def dissipation(state: State, time: float) -> torch.Tensor:
        return diagnostics.dissipation(mesh, state["velocity"], VISCOSITY)

    sampled = {"spectrum": spectrum, "dissipation": dissipation}
    return Outputs(path, INTERVAL, {"energy": energy}, sampled=sampled)


def random_outputs(path: Path) -> Outputs:
    generator = torch.Generator().manual_seed(0)

    def noise(state: State, time: float) -> torch.Tensor:
        return torch.randn(COMPONENTS, generator=generator, dtype=torch.float64)

    return Outputs(path, INTERVAL, sampled={"noise": noise})


def read_dataset(file: h5py.File, name: str) -> torch.Tensor:
    dataset = file[name]
    assert isinstance(dataset, h5py.Dataset)
    return torch.as_tensor(dataset[()])


def group_names(file: h5py.File, name: str) -> list[str]:
    group = file[name]
    assert isinstance(group, h5py.Group)
    return sorted(str(key) for key in group)


def read_time_series(directory: Path) -> dict[str, torch.Tensor]:
    with h5py.File(directory / "time_series.h5", "r") as series:
        return {str(name): read_dataset(series, str(name)) for name in series}


def read_sampled(directory: Path, name: str) -> tuple[list[float], torch.Tensor]:
    times: list[float] = []
    values: list[torch.Tensor] = []
    for path in sorted(directory.glob("snapshot_*.h5")):
        with h5py.File(path, "r") as snapshot:
            times.append(float(torch.as_tensor(snapshot.attrs["time"]).item()))
            values.append(read_dataset(snapshot, f"diagnostics/{name}"))
    return times, torch.stack(values)


def block_standard_error(samples: torch.Tensor, block_length: int) -> torch.Tensor:
    blocks = samples.shape[0] // block_length
    used = samples[: blocks * block_length]
    means = used.reshape(blocks, block_length, *samples.shape[1:]).mean(dim=1)
    return means.std(dim=0, correction=1) / math.sqrt(blocks)


def test_the_window_means_of_sampled_diagnostics_are_their_analytic_time_averages(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    run(mesh, spectrum_outputs(mesh, tmp_path / "run"), 1)
    statistic = tmp_path / "time_average.h5"

    window_statistic(tmp_path / "run", statistic, 0.25, 1, block_length=2)

    times = [0.25, 0.5, 0.75, 1.0]
    expected = torch.stack([exact_spectrum(time) for time in times]).mean(dim=0)
    with h5py.File(statistic, "r") as average:
        spectrum = read_dataset(average, "mean/spectrum")
        spectrum_error = read_dataset(average, "standard_error/spectrum")
        dissipation = read_dataset(average, "mean/dissipation")
        dissipation_error = read_dataset(average, "standard_error/dissipation")
    torch.testing.assert_close(spectrum, expected, rtol=0, atol=1e-14)
    assert spectrum_error.shape == expected.shape
    assert dissipation.item() == pytest.approx(
        sum(exact_dissipation(time) for time in times) / len(times), rel=1e-12
    )
    assert dissipation_error.shape == ()


def test_a_trailing_partial_block_is_left_out_of_the_standard_error_only(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    run(mesh, random_outputs(tmp_path / "run"), 2.25)
    statistic = tmp_path / "time_average.h5"

    window_statistic(tmp_path / "run", statistic, 0, 2.25, block_length=3)

    times, samples = read_sampled(tmp_path / "run", "noise")
    assert len(times) == 10
    with h5py.File(statistic, "r") as average:
        mean = read_dataset(average, "mean/noise")
        standard_error = read_dataset(average, "standard_error/noise")
    torch.testing.assert_close(mean, samples.mean(dim=0), rtol=1e-14, atol=0)
    torch.testing.assert_close(
        standard_error, block_standard_error(samples, 3), rtol=1e-14, atol=0
    )


def test_the_standard_error_of_independent_samples_matches_their_variance(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    run(mesh, random_outputs(tmp_path / "run"), 2)
    statistic = tmp_path / "time_average.h5"

    window_statistic(tmp_path / "run", statistic, 0.25, 2, block_length=2)

    with h5py.File(statistic, "r") as average:
        standard_error = read_dataset(average, "standard_error/noise")
    samples = 8
    assert standard_error.square().mean().item() == pytest.approx(1 / samples, rel=0.05)


def test_time_series_columns_get_a_mean_and_a_standard_error_over_the_window(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    run(mesh, spectrum_outputs(mesh, tmp_path / "run"), 1)
    statistic = tmp_path / "time_average.h5"

    window_statistic(tmp_path / "run", statistic, 0.25, 1, block_length=2)

    series = read_time_series(tmp_path / "run")
    inside = (series["time"] >= 0.25) & (series["time"] <= 1)
    times = series["time"][inside].tolist()
    exact = torch.tensor([exact_energy(time) for time in times], dtype=torch.float64)
    with h5py.File(statistic, "r") as average:
        mean = read_dataset(average, "mean/energy")
        standard_error = read_dataset(average, "standard_error/energy")
        time_series_samples = average.attrs["time_series_samples"]
    assert time_series_samples == len(times) == 10
    assert mean.item() == pytest.approx(exact.mean().item(), rel=1e-12)
    assert standard_error.item() == pytest.approx(
        block_standard_error(exact, 2).item(), rel=1e-10
    )


def test_the_time_average_file_records_its_window_shells_and_mesh(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    shells = Shells(mesh)
    run(mesh, spectrum_outputs(mesh, tmp_path / "run"), 1)
    statistic = tmp_path / "time_average.h5"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    window_statistic(tmp_path / "run", statistic, 0.25, 1, block_length=2)

    with h5py.File(statistic, "r") as average:
        assert sorted(average) == ["mean", "shells", "standard_error"]
        names = ["dissipation", "energy", "spectrum"]
        assert group_names(average, "mean") == names
        assert group_names(average, "standard_error") == names
        assert torch.equal(
            read_dataset(average, "shells/wavenumbers"), shells.wavenumbers
        )
        assert torch.equal(read_dataset(average, "shells/counts"), shells.counts)
        assert dict(average.attrs) == {
            "start": 0.25,
            "end": 1.0,
            "snapshot_samples": 4,
            "time_series_samples": 10,
            "block_length": 2,
            "points": POINTS,
            "length": LENGTH,
            "dimension": 1,
            "dtype": "float64",
            "commit": head,
        }


def test_the_statistic_refuses_to_overwrite_a_file(tmp_path: Path) -> None:
    mesh = make_mesh()
    run(mesh, spectrum_outputs(mesh, tmp_path / "run"), 1)
    statistic = tmp_path / "time_average.h5"
    statistic.write_bytes(b"earlier window")

    with pytest.raises(FileExistsError):
        window_statistic(tmp_path / "run", statistic, 0.25, 1, block_length=2)

    assert statistic.read_bytes() == b"earlier window"


@pytest.mark.parametrize(
    ("start", "end", "block_length"), [(0.25, 1, 3), (0.5, 1, 2), (2, 3, 1)]
)
def test_fewer_than_two_blocks_raise(
    tmp_path: Path, start: float, end: float, block_length: int
) -> None:
    mesh = make_mesh()
    run(mesh, spectrum_outputs(mesh, tmp_path / "run"), 1)
    statistic = tmp_path / "time_average.h5"

    with pytest.raises(ValueError, match="two blocks"):
        window_statistic(tmp_path / "run", statistic, start, end, block_length)

    assert not statistic.exists()


def test_a_block_length_below_one_raises(tmp_path: Path) -> None:
    mesh = make_mesh()
    run(mesh, spectrum_outputs(mesh, tmp_path / "run"), 1)

    with pytest.raises(ValueError, match="block length"):
        window_statistic(tmp_path / "run", tmp_path / "average.h5", 0, 1, 0)

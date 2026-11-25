import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import h5py
import pytest
import torch

from specflow import diagnostics, flows
from specflow.dealiasing import TwoThirdsDealiasing
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4
from specflow.mesh import Mesh
from specflow.outputs import TIME_SERIES_NAME, Outputs
from specflow.simulation import Simulation
from specflow.terms import Advection, Diffusion, State
from specflow.time_step import CFLTimeStep

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "TGV_Re1600.dat"
CASE = ROOT / "cases" / "taylor_green_3d.py"
CASE_POINTS = 16
CASE_SNAPSHOTS = 5
SAMPLED = (
    "advection_budget_spectrum",
    "diffusion_budget_spectrum",
    "energy_spectrum",
    "longitudinal_derivative_counts",
    "longitudinal_derivative_moments",
)
VISCOSITY = 1 / 1600
LENGTH = 2 * math.pi
CFL_TARGET = 0.5
INITIAL_TIME_STEP = 1e-3
DIAGNOSTICS = ("energy", "dissipation", "enstrophy")
END = 10.0
PEAK_DISSIPATION = 1.28486e-2
PEAK_DISSIPATION_TOLERANCE = 0.03
PEAK_TIME = 8.98
PEAK_TIME_TOLERANCE = 0.2
RESOLVED_END = 7.0
RESOLVED_TOLERANCE = 0.01


@pytest.fixture(scope="module")
def case_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("taylor_green_3d") / "output"
    subprocess.run(
        [sys.executable, str(CASE), str(CASE_POINTS), "--output", str(path)],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture
def cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return torch.device("cuda")


def read_reference() -> dict[str, torch.Tensor]:
    rows = [
        [float(value) for value in line.split()]
        for line in REFERENCE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    columns = torch.tensor(rows, dtype=torch.float64).T
    return {
        "time": columns[0],
        "energy": columns[1],
        "dissipation": columns[3],
        "enstrophy": columns[4],
    }


def run_taylor_green_3d(
    points: int, end: float, path: Path, device: torch.device
) -> dict[str, torch.Tensor]:
    mesh = Mesh(points, LENGTH, 3, SerialDecomposition(points, 3), device=device)
    terms = [
        Diffusion(mesh, "velocity", VISCOSITY, components=(3,)),
        Advection(mesh, TwoThirdsDealiasing(mesh), "velocity"),
    ]
    equation = Equation("velocity", terms, projection=mesh.project)

    def energy(state: State, time: float) -> torch.Tensor:
        return diagnostics.kinetic_energy(mesh, state["velocity"])

    def dissipation(state: State, time: float) -> torch.Tensor:
        return diagnostics.dissipation(mesh, state["velocity"], VISCOSITY)

    def enstrophy(state: State, time: float) -> torch.Tensor:
        return diagnostics.enstrophy(mesh, state["velocity"])

    named = {"energy": energy, "dissipation": dissipation, "enstrophy": enstrophy}
    outputs = Outputs(path, END, named)
    time_step = CFLTimeStep("velocity", CFL_TARGET, INITIAL_TIME_STEP)
    simulation = Simulation(mesh, [equation], IFRK4(), time_step, outputs)
    velocity_physical = flows.taylor_green_3d_physical(mesh)
    state = {"velocity": mesh.decomposition.to_spectral(velocity_physical)}

    simulation.run(state, 0.0, end)

    with h5py.File(path / TIME_SERIES_NAME, "r") as series:
        return {name: read_dataset(series, name) for name in ("time", *DIAGNOSTICS)}


def read_dataset(file: h5py.File, name: str) -> torch.Tensor:
    dataset = file[name]
    assert isinstance(dataset, h5py.Dataset)
    return torch.from_numpy(dataset[()])


def peak(time: torch.Tensor, values: torch.Tensor) -> tuple[float, float]:
    index = int(values.argmax().item())
    return time[index].item(), values[index].item()


def interpolate(
    time: torch.Tensor, reference_time: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    upper = torch.searchsorted(reference_time, time).clamp(1, len(reference_time) - 1)
    lower = upper - 1
    weight = (time - reference_time[lower]) / (
        reference_time[upper] - reference_time[lower]
    )
    return torch.lerp(reference[lower], reference[upper], weight)


def resolved_error(time: torch.Tensor, dissipation: torch.Tensor) -> float:
    reference = read_reference()
    resolved = time <= RESOLVED_END
    expected = interpolate(time[resolved], reference["time"], reference["dissipation"])
    return (dissipation[resolved] / expected - 1).abs().max().item()


def assert_matches_reference_curve(
    time: torch.Tensor, dissipation: torch.Tensor
) -> None:
    assert resolved_error(time, dissipation) <= RESOLVED_TOLERANCE


def assert_matches_reference_peak(
    time: torch.Tensor, dissipation: torch.Tensor
) -> None:
    peak_time, peak_dissipation = peak(time, dissipation)
    assert peak_dissipation == pytest.approx(
        PEAK_DISSIPATION, rel=PEAK_DISSIPATION_TOLERANCE
    )
    assert peak_time == pytest.approx(PEAK_TIME, abs=PEAK_TIME_TOLERANCE)


def test_initial_diagnostics_are_the_first_reference_row(tmp_path: Path) -> None:
    reference = read_reference()

    series = run_taylor_green_3d(16, 0.0, tmp_path, torch.device("cpu"))

    assert series["time"].shape == (1,)
    assert series["time"][0].item() == 0.0
    for name in DIAGNOSTICS:
        assert series[name][0].item() == pytest.approx(
            reference[name][0].item(), rel=1e-12
        )


def test_initial_dissipation_is_three_quarters_of_the_viscosity(
    tmp_path: Path,
) -> None:
    series = run_taylor_green_3d(16, 0.0, tmp_path, torch.device("cpu"))

    assert series["energy"][0].item() == pytest.approx(0.125, rel=1e-12)
    assert series["dissipation"][0].item() == pytest.approx(0.75 * VISCOSITY, rel=1e-12)
    assert series["enstrophy"][0].item() == pytest.approx(0.375, rel=1e-12)


def test_reference_dissipation_passes_both_checks() -> None:
    reference = read_reference()

    assert_matches_reference_curve(reference["time"], reference["dissipation"])
    assert_matches_reference_peak(reference["time"], reference["dissipation"])


@pytest.mark.parametrize("scale", [0.989, 1.011])
def test_curve_check_rejects_a_curve_off_by_more_than_one_percent(
    scale: float,
) -> None:
    reference = read_reference()

    with pytest.raises(AssertionError):
        assert_matches_reference_curve(
            reference["time"], reference["dissipation"] * scale
        )


def test_curve_check_ignores_errors_after_the_resolved_end() -> None:
    reference = read_reference()
    scale = torch.where(reference["time"] > RESOLVED_END, 1.1, 1.0)

    assert_matches_reference_curve(reference["time"], reference["dissipation"] * scale)


def test_curve_check_compares_at_the_simulated_times() -> None:
    reference = read_reference()
    time = torch.linspace(0.0, RESOLVED_END, 71, dtype=torch.float64)
    dissipation = interpolate(time, reference["time"], reference["dissipation"])

    assert_matches_reference_curve(time, dissipation * 1.009)


@pytest.mark.parametrize("scale", [0.969, 1.031])
def test_peak_check_rejects_a_peak_off_by_more_than_three_percent(
    scale: float,
) -> None:
    reference = read_reference()

    with pytest.raises(AssertionError):
        assert_matches_reference_peak(
            reference["time"], reference["dissipation"] * scale
        )


@pytest.mark.parametrize("shift", [-0.21, 0.21])
def test_peak_check_rejects_a_peak_off_by_more_than_the_time_tolerance(
    shift: float,
) -> None:
    reference = read_reference()

    with pytest.raises(AssertionError):
        assert_matches_reference_peak(
            reference["time"] + shift, reference["dissipation"]
        )


@pytest.mark.parametrize("scale", [0.975, 1.025])
def test_peak_check_accepts_a_peak_within_the_tolerances(scale: float) -> None:
    reference = read_reference()

    assert_matches_reference_peak(
        reference["time"] - 0.18, reference["dissipation"] * scale
    )


@pytest.mark.gpu
@pytest.mark.parametrize("points", [64, 128])
def test_coarser_meshes_against_the_reference(
    points: int,
    cuda: torch.device,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    series = run_taylor_green_3d(points, END, tmp_path, cuda)

    peak_time, peak_dissipation = peak(series["time"], series["dissipation"])
    record_property("peak_time", peak_time)
    record_property("peak_dissipation", peak_dissipation)
    record_property("peak_time_error", peak_time - PEAK_TIME)
    record_property("peak_dissipation_error", peak_dissipation - PEAK_DISSIPATION)
    record_property(
        "resolved_error", resolved_error(series["time"], series["dissipation"])
    )


@pytest.mark.gpu
def test_dissipation_at_256_matches_the_reference(
    cuda: torch.device, tmp_path: Path
) -> None:
    series = run_taylor_green_3d(256, END, tmp_path, cuda)

    assert_matches_reference_curve(series["time"], series["dissipation"])
    assert_matches_reference_peak(series["time"], series["dissipation"])


def read_snapshots(path: Path) -> list[dict[str, torch.Tensor]]:
    snapshots: list[dict[str, torch.Tensor]] = []
    for snapshot_path in sorted(path.glob("snapshot_*.h5")):
        with h5py.File(snapshot_path, "r") as snapshot:
            group = snapshot["diagnostics"]
            assert isinstance(group, h5py.Group)
            names = [f"diagnostics/{key}" for key in group]
            names += ["shells/wavenumbers", "shells/counts"]
            entries = {name: read_dataset(snapshot, name) for name in names}
            entries["step"] = torch.as_tensor(snapshot.attrs["step"])
        snapshots.append(entries)
    return snapshots


def read_series_rows(path: Path, steps: torch.Tensor) -> dict[str, torch.Tensor]:
    with h5py.File(path / TIME_SERIES_NAME, "r") as series:
        columns = {name: read_dataset(series, name) for name in ("step", *DIAGNOSTICS)}
    rows = torch.searchsorted(columns["step"], steps)
    assert torch.equal(columns["step"][rows], steps)
    return {name: column[rows] for name, column in columns.items()}


def test_the_case_writes_its_sampled_diagnostics_into_each_snapshot(
    case_output: Path,
) -> None:
    snapshots = read_snapshots(case_output)

    assert len(snapshots) == CASE_SNAPSHOTS
    for snapshot in snapshots:
        stored = sorted(key for key in snapshot if key.startswith("diagnostics/"))
        assert stored == [f"diagnostics/{name}" for name in SAMPLED]
        shells = snapshot["shells/wavenumbers"].shape
        for name in SAMPLED:
            if name.endswith("spectrum"):
                assert snapshot[f"diagnostics/{name}"].shape == shells
        assert snapshot["diagnostics/longitudinal_derivative_moments"].shape == (4, 3)
        assert snapshot["shells/counts"].sum().item() == CASE_POINTS**3


def test_the_sampled_spectra_close_on_the_time_series(case_output: Path) -> None:
    snapshots = read_snapshots(case_output)
    steps = torch.stack([snapshot["step"] for snapshot in snapshots])
    rows = read_series_rows(case_output, steps)

    width = 2 * math.pi / LENGTH
    for index, snapshot in enumerate(snapshots):
        energy = snapshot["diagnostics/energy_spectrum"].sum().item() * width
        diffusion = (
            snapshot["diagnostics/diffusion_budget_spectrum"].sum().item() * width
        )
        advection = (
            snapshot["diagnostics/advection_budget_spectrum"].sum().item() * width
        )
        assert energy == pytest.approx(rows["energy"][index].item(), rel=1e-12)
        assert diffusion == pytest.approx(-rows["dissipation"][index].item(), rel=1e-12)
        assert advection == pytest.approx(0.0, abs=1e-14)


def test_the_sampled_counts_hold_every_point_of_each_component(
    case_output: Path,
) -> None:
    for snapshot in read_snapshots(case_output):
        counts = snapshot["diagnostics/longitudinal_derivative_counts"]
        assert counts.dtype == torch.int64
        assert counts.sum(-1).tolist() == [CASE_POINTS**3] * 3


def test_the_initial_moments_of_the_longitudinal_derivatives(
    case_output: Path,
) -> None:
    moments = read_snapshots(case_output)[0][
        "diagnostics/longitudinal_derivative_moments"
    ]

    assert moments[0].tolist() == pytest.approx([0.0] * 3, abs=1e-15)
    assert moments[1].tolist() == pytest.approx([0.125, 0.125, 0.0], abs=1e-15)
    assert moments[2].tolist() == pytest.approx([0.0] * 3, abs=1e-15)
    assert moments[3].tolist() == pytest.approx([(3 / 8) ** 3] * 2 + [0.0], abs=1e-15)

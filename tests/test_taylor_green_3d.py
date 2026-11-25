import math
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

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "TGV_Re1600.dat"
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

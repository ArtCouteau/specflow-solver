import math
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
    outputs = Outputs(path, max(end, 1.0), named)
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


def test_initial_diagnostics_are_the_first_reference_row(tmp_path: Path) -> None:
    reference = read_reference()

    series = run_taylor_green_3d(16, 0.0, tmp_path, torch.device("cpu"))

    assert series["time"].tolist() == [0.0]
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

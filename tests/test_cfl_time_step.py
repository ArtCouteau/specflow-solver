import math
from pathlib import Path

import torch

from specflow import flows
from specflow.dealiasing import TwoThirdsDealiasing
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4
from specflow.mesh import Mesh
from specflow.outputs import Outputs, Snapshot, load_snapshot
from specflow.simulation import Simulation
from specflow.terms import Advection, Diffusion
from specflow.time_step import CFLTimeStep

POINTS = 16
LENGTH = 1.0
VISCOSITY = 0.025
MEAN_FLOW = (0.5, 0.25)
TARGET = 0.45
INITIAL = 1e-3
INTERVAL = 0.25


def make_mesh() -> Mesh:
    return Mesh(POINTS, LENGTH, 2, SerialDecomposition(POINTS, 2))


def make_simulation(
    mesh: Mesh, outputs: Outputs | None, initial: float = INITIAL
) -> Simulation:
    terms = [
        Diffusion(mesh, "velocity", VISCOSITY, components=(2,)),
        Advection(mesh, TwoThirdsDealiasing(mesh), "velocity"),
    ]
    equation = Equation("velocity", terms, projection=mesh.project)
    time_step = CFLTimeStep("velocity", TARGET, initial)
    return Simulation(mesh, [equation], IFRK4(), time_step, outputs)


def taylor_green_with_mean_flow(mesh: Mesh) -> dict[str, torch.Tensor]:
    velocity_physical = flows.taylor_green_2d_physical(mesh, VISCOSITY, 0.0)
    mean = torch.tensor(MEAN_FLOW, dtype=mesh.dtype).reshape(2, 1, 1)
    return {"velocity": mesh.decomposition.to_spectral(velocity_physical + mean)}


def cfl_number(mesh: Mesh, snapshot: Snapshot) -> float:
    velocity_physical = mesh.decomposition.to_physical(snapshot.state["velocity"])
    speed = velocity_physical.abs().sum(dim=0).max().item()
    return snapshot.time_step * speed / mesh.spacing


def snapshots(mesh: Mesh, directory: Path) -> list[Snapshot]:
    return [load_snapshot(p, mesh) for p in sorted(directory.glob("snapshot_*.h5"))]


def test_the_cfl_number_reaches_its_target(tmp_path: Path) -> None:
    mesh = make_mesh()

    make_simulation(mesh, Outputs(tmp_path, INTERVAL)).run(
        taylor_green_with_mean_flow(mesh), 0, 1
    )

    written = snapshots(mesh, tmp_path)
    assert written[0].time_step == INITIAL
    for snapshot in written[1:]:
        assert math.isclose(cfl_number(mesh, snapshot), TARGET, rel_tol=1e-12)

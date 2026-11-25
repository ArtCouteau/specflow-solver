import math
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from specflow import flows
from specflow.dealiasing import TwoThirdsDealiasing
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4
from specflow.mesh import Mesh
from specflow.outputs import Outputs, Snapshot, load_snapshot
from specflow.simulation import Simulation
from specflow.terms import Advection, Diffusion, State
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


def uniform_flow(mesh: Mesh, velocity: tuple[float, float]) -> dict[str, torch.Tensor]:
    velocity_physical = torch.tensor(velocity, dtype=mesh.dtype).reshape(2, 1, 1)
    velocity_physical = velocity_physical.expand(2, *mesh.physical_shape)
    return {"velocity": mesh.decomposition.to_spectral(velocity_physical)}


def test_shortened_steps_never_become_the_controllers_time_step(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()
    speed = sum(abs(component) for component in MEAN_FLOW) / mesh.spacing

    make_simulation(mesh, Outputs(tmp_path, INTERVAL)).run(
        uniform_flow(mesh, MEAN_FLOW), 0, 1
    )

    written = snapshots(mesh, tmp_path)
    assert [snapshot.time for snapshot in written] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [snapshot.step for snapshot in written] == [0, 22, 29, 36, 43]
    for snapshot in written:
        expected = min(INITIAL * 1.2**snapshot.step, TARGET / speed)
        assert math.isclose(snapshot.time_step, expected, rel_tol=1e-12)


def test_a_fluid_at_rest_grows_the_time_step_by_the_maximum_growth(
    tmp_path: Path,
) -> None:
    mesh = make_mesh()

    make_simulation(mesh, Outputs(tmp_path, INTERVAL)).run(
        uniform_flow(mesh, (0.0, 0.0)), 0, 0.5
    )

    for snapshot in snapshots(mesh, tmp_path):
        expected = INITIAL * 1.2**snapshot.step
        assert math.isclose(snapshot.time_step, expected, rel_tol=1e-12)


def test_a_resumed_run_reproduces_the_uninterrupted_run(tmp_path: Path) -> None:
    mesh = make_mesh()
    initial = taylor_green_with_mean_flow(mesh)
    interval = 0.05
    whole_outputs = Outputs(tmp_path / "whole", interval)
    make_simulation(mesh, whole_outputs).run(initial, 0, 0.25)
    make_simulation(mesh, Outputs(tmp_path / "split", interval)).run(initial, 0, 0.05)

    outputs = Outputs(tmp_path / "split", interval, resume=True)
    latest = outputs.latest_snapshot(mesh)
    simulation = make_simulation(mesh, outputs, initial=latest.time_step)
    final = simulation.run(latest.state, latest.time, 0.25)

    whole = snapshots(mesh, tmp_path / "whole")
    split = snapshots(mesh, tmp_path / "split")
    assert cfl_number(mesh, latest) < TARGET
    assert [(s.time, s.step, s.time_step) for s in split] == [
        (s.time, s.step, s.time_step) for s in whole
    ]
    assert torch.equal(final["velocity"], whole[-1].state["velocity"])


class BlowingUp:
    def __init__(self, after: float) -> None:
        self.after = after

    @property
    def name(self) -> str:
        return "blowing up"

    @property
    def reads(self) -> Mapping[str, tuple[int, ...]]:
        return {"velocity": (2,)}

    def contribution(self, state: State, time: float) -> torch.Tensor:
        value = math.nan if time > self.after else 0.0
        return torch.full_like(state["velocity"], value)


def test_a_non_finite_cfl_number_raises_with_its_step_and_time() -> None:
    mesh = make_mesh()
    speed = sum(abs(component) for component in MEAN_FLOW) / mesh.spacing
    equation = Equation("velocity", [BlowingUp(after=0.31)], projection=mesh.project)
    time_step = CFLTimeStep("velocity", TARGET, TARGET / speed)
    simulation = Simulation(mesh, [equation], IFRK4(), time_step)

    with pytest.raises(FloatingPointError, match="CFL number.*step 9, time 0.3375"):
        simulation.run(uniform_flow(mesh, MEAN_FLOW), 0, 1)

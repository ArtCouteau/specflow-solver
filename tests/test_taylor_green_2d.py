import math

import torch

from specflow import flows
from specflow.dealiasing import TwoThirdsDealiasing
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4, RK4, Integrator
from specflow.mesh import Mesh
from specflow.simulation import Simulation
from specflow.terms import Advection, Diffusion
from specflow.time_step import FixedTimeStep

POINTS = 16
LENGTH = 1.0
VISCOSITY = 0.025


def make_mesh() -> Mesh:
    return Mesh(POINTS, LENGTH, 2, SerialDecomposition(POINTS, 2))


def exact_spectral(mesh: Mesh, time: float) -> torch.Tensor:
    velocity_physical = flows.taylor_green_2d_physical(mesh, VISCOSITY, time)
    return mesh.decomposition.to_spectral(velocity_physical)


def norm(mesh: Mesh, field_spectral: torch.Tensor) -> float:
    squares = field_spectral.abs().square() * mesh.parseval_weights
    return math.sqrt(mesh.decomposition.sum(squares).sum().item())


def run(
    integrator: Integrator, time_step: float, start: float, end: float
) -> tuple[float, float]:
    mesh = make_mesh()
    terms = [
        Diffusion(mesh, "velocity", VISCOSITY, components=(2,)),
        Advection(mesh, TwoThirdsDealiasing(mesh), "velocity"),
    ]
    equation = Equation("velocity", terms, projection=mesh.project)
    simulation = Simulation(mesh, [equation], integrator, FixedTimeStep(time_step))
    state = {"velocity": exact_spectral(mesh, start)}

    final = simulation.run(state, start, end)

    expected_spectral = exact_spectral(mesh, end)
    error = norm(mesh, final["velocity"] - expected_spectral)
    divergence = norm(mesh, mesh.divergence(final["velocity"]))
    return error / norm(mesh, expected_spectral), divergence


def test_ifrk4_is_exact_to_round_off() -> None:
    error, divergence = run(IFRK4(), 0.03, 0.2, 1.0)

    assert error <= 1e-12
    assert divergence <= 1e-12


def test_rk4_converges_at_fourth_order() -> None:
    coarse, _ = run(RK4(), 0.02, 0.0, 0.6)
    fine, divergence = run(RK4(), 0.01, 0.0, 0.6)

    assert abs(math.log2(coarse / fine) - 4) <= 0.1
    assert divergence <= 1e-12

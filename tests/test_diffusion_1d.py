import math

import torch

from specflow import flows
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4, RK4, Integrator
from specflow.mesh import Mesh
from specflow.simulation import Simulation
from specflow.terms import Diffusion
from specflow.time_step import FixedTimeStep

POINTS = 16
DIFFUSIVITY = 0.5
AMPLITUDES = {1: 1.0, 2: -0.5, 3: 0.25}


def make_mesh() -> Mesh:
    return Mesh(POINTS, 2 * math.pi, 1, SerialDecomposition(POINTS, 1))


def exact_spectral(mesh: Mesh, time: float) -> torch.Tensor:
    solution_physical = flows.diffusion_1d_physical(mesh, AMPLITUDES, DIFFUSIVITY, time)
    return mesh.decomposition.to_spectral(solution_physical)


def norm(field_spectral: torch.Tensor) -> float:
    return field_spectral.abs().square().sum().sqrt().item()


def run(integrator: Integrator, time_step: float, start: float, end: float) -> float:
    mesh = make_mesh()
    equation = Equation("scalar", [Diffusion(mesh, "scalar", DIFFUSIVITY)])
    simulation = Simulation(mesh, [equation], integrator, FixedTimeStep(time_step))
    state = {"scalar": exact_spectral(mesh, start)}

    final = simulation.run(state, start, end)

    expected_spectral = exact_spectral(mesh, end)
    return norm(final["scalar"] - expected_spectral) / norm(expected_spectral)


def test_ifrk4_is_exact_to_round_off() -> None:
    assert run(IFRK4(), 0.03, 0.2, 1.0) <= 1e-12


def test_rk4_converges_at_fourth_order() -> None:
    coarse = run(RK4(), 0.02, 0.0, 0.8)
    fine = run(RK4(), 0.01, 0.0, 0.8)

    assert abs(math.log2(coarse / fine) - 4) <= 0.1

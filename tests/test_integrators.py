import math
from collections.abc import Mapping

import pytest
import torch

from specflow import flows
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4, RK4, Integrator
from specflow.mesh import Mesh
from specflow.simulation import Simulation
from specflow.terms import Diffusion, State
from specflow.time_step import FixedTimeStep

POINTS = 16
DIFFUSIVITY = 0.5
FREQUENCY = 3.0
AMPLITUDES = {1: 1.0, 2: -0.5, 3: 0.25}


class Rotation:
    @property
    def name(self) -> str:
        return "rotation"

    @property
    def reads(self) -> Mapping[str, tuple[int, ...]]:
        return {"scalar": ()}

    def contribution(self, state: State, time: float) -> torch.Tensor:
        return state["scalar"] * (1j * FREQUENCY)


def norm(field_spectral: torch.Tensor) -> float:
    return field_spectral.abs().square().sum().sqrt().item()


def relative_error(integrator: Integrator, time_step: float) -> float:
    mesh = Mesh(POINTS, 2 * math.pi, 1, SerialDecomposition(POINTS, 1))
    terms = [Diffusion(mesh, "scalar", DIFFUSIVITY), Rotation()]
    equation = Equation("scalar", terms)
    simulation = Simulation(mesh, [equation], integrator, FixedTimeStep(time_step))
    initial_physical = flows.diffusion_1d_physical(mesh, AMPLITUDES, DIFFUSIVITY, 0.0)
    initial_spectral = mesh.decomposition.to_spectral(initial_physical)
    end = 0.8

    final = simulation.run({"scalar": initial_spectral}, 0.0, end)

    growth = (-DIFFUSIVITY * mesh.wavenumbers_squared + 1j * FREQUENCY) * end
    expected_spectral = initial_spectral * growth.exp()
    return norm(final["scalar"] - expected_spectral) / norm(expected_spectral)


@pytest.mark.parametrize("integrator", [RK4(), IFRK4()], ids=["RK4", "IFRK4"])
def test_a_nonlinear_term_converges_at_fourth_order(integrator: Integrator) -> None:
    coarse = relative_error(integrator, 0.02)
    fine = relative_error(integrator, 0.01)

    assert abs(math.log2(coarse / fine) - 4) <= 0.1

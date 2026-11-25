"""The decaying 2D Taylor-Green vortex, checked against its exact solution.

Usage: ``python cases/taylor_green_2d.py [--output DIRECTORY]``.
"""

import argparse
import logging
import math
from pathlib import Path

import torch

from specflow import diagnostics, flows
from specflow.dealiasing import TwoThirdsDealiasing
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4
from specflow.mesh import Mesh
from specflow.outputs import Outputs
from specflow.simulation import Simulation
from specflow.terms import Advection, Diffusion, State
from specflow.time_step import CFLTimeStep

POINTS = 64
LENGTH = 2 * math.pi
VISCOSITY = 0.01
CFL_TARGET = 0.5
INITIAL_TIME_STEP = 1e-3
SNAPSHOT_INTERVAL = 1.0
END = 5.0

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the case and log the relative L2 error against the exact solution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("taylor_green_2d"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    mesh = Mesh(POINTS, LENGTH, 2, SerialDecomposition(POINTS, 2))
    terms = [
        Diffusion(mesh, "velocity", VISCOSITY, components=(2,)),
        Advection(mesh, TwoThirdsDealiasing(mesh), "velocity"),
    ]
    equation = Equation("velocity", terms, projection=mesh.project)

    def exact_spectral(time: float) -> torch.Tensor:
        velocity_physical = flows.taylor_green_2d_physical(mesh, VISCOSITY, time)
        return mesh.decomposition.to_spectral(velocity_physical)

    def error(state: State, time: float) -> torch.Tensor:
        return diagnostics.relative_error(
            mesh, state["velocity"], exact_spectral(time), component_axes=1
        )

    def divergence(state: State, time: float) -> torch.Tensor:
        return diagnostics.divergence_rms(mesh, state["velocity"])

    outputs = Outputs(
        arguments.output,
        SNAPSHOT_INTERVAL,
        {"error": error, "divergence": divergence},
    )
    time_step = CFLTimeStep("velocity", CFL_TARGET, INITIAL_TIME_STEP)
    simulation = Simulation(mesh, [equation], IFRK4(), time_step, outputs)
    final = simulation.run({"velocity": exact_spectral(0.0)}, 0.0, END)
    logger.info(f"relative error at t = {END:g}: {error(final, END).item():.3e}")


if __name__ == "__main__":
    main()

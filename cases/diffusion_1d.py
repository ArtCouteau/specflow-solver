"""Multi-mode 1D diffusion, integrated exactly by IFRK4 and checked at the end.

Usage: ``python cases/diffusion_1d.py [--output DIRECTORY]``.
"""

import argparse
import logging
import math
from pathlib import Path

import torch

from specflow import diagnostics, flows
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4
from specflow.mesh import Mesh
from specflow.outputs import Outputs
from specflow.simulation import Simulation
from specflow.terms import Diffusion, State
from specflow.time_step import FixedTimeStep

POINTS = 32
LENGTH = 2 * math.pi
DIFFUSIVITY = 0.1
AMPLITUDES = {1: 1.0, 2: -0.5, 3: 0.25, 5: 0.125}
TIME_STEP = 0.05
SNAPSHOT_INTERVAL = 0.5
END = 2.0

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the case and log the relative L2 error against the exact solution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("diffusion_1d"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    mesh = Mesh(POINTS, LENGTH, 1, SerialDecomposition(POINTS, 1))
    equation = Equation("scalar", [Diffusion(mesh, "scalar", DIFFUSIVITY)])

    def exact_spectral(time: float) -> torch.Tensor:
        solution_physical = flows.diffusion_1d_physical(
            mesh, AMPLITUDES, DIFFUSIVITY, time
        )
        return mesh.decomposition.to_spectral(solution_physical)

    def error(state: State, time: float) -> torch.Tensor:
        return diagnostics.relative_error(mesh, state["scalar"], exact_spectral(time))

    outputs = Outputs(arguments.output, SNAPSHOT_INTERVAL, {"error": error})
    simulation = Simulation(
        mesh, [equation], IFRK4(), FixedTimeStep(TIME_STEP), outputs
    )
    final = simulation.run({"scalar": exact_spectral(0.0)}, 0.0, END)
    logger.info("relative error at t = %g: %.3e", END, error(final, END).item())


if __name__ == "__main__":
    main()

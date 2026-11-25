"""The 3D Taylor-Green vortex at Re 1600, compared with the Incompact3d reference.

The reference is the $512^3$ DNS of Dairay, Lamballais, Laizet and Vassilicos,
J. Comput. Phys. 337 (2017) 252-274, in ``data/TGV_Re1600.dat``. The run writes
the kinetic energy, dissipation and enstrophy at every step to the time series of
its output directory, ``taylor_green_3d_<N>`` by default, and logs its peak
dissipation against the reference's at the end. Each snapshot also holds the
energy spectrum, the budget spectra of advection and diffusion, and the raw
moments and histogram counts of the longitudinal derivatives; the flow decays, so
their window statistic only shows the wiring.

Usage: ``python cases/taylor_green_3d.py N [--output DIRECTORY] [--resume]``.
"""

import argparse
import logging
import math
from pathlib import Path

import h5py
import torch

from specflow import diagnostics, distributions, flows, spectra
from specflow.dealiasing import TwoThirdsDealiasing
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4
from specflow.mesh import Mesh
from specflow.outputs import TIME_SERIES_NAME, Diagnostic, Outputs
from specflow.simulation import Simulation
from specflow.spectra import Shells
from specflow.terms import Advection, Diffusion, State
from specflow.time_step import CFLTimeStep

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "TGV_Re1600.dat"
LENGTH = 2 * math.pi
VISCOSITY = 1 / 1600
CFL_TARGET = 0.5
INITIAL_TIME_STEP = 1e-3
SNAPSHOT_INTERVAL = 5.0
END = 20.0
DERIVATIVE_BOUND = 20.0
DERIVATIVE_BINS = 200

logger = logging.getLogger(__name__)


def reference_peak() -> tuple[float, float]:
    """Return the time and value of the reference's peak dissipation."""
    rows = [
        [float(value) for value in line.split()]
        for line in REFERENCE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    columns = torch.tensor(rows, dtype=torch.float64).T
    return peak(columns[0], columns[3])


def peak(time: torch.Tensor, values: torch.Tensor) -> tuple[float, float]:
    """Return the time and value of the largest of a series of values."""
    index = int(values.argmax().item())
    return time[index].item(), values[index].item()


def series_peak(path: Path, name: str) -> tuple[float, float]:
    """Return the time and value of the peak of a diagnostic of a time series."""
    with h5py.File(path / TIME_SERIES_NAME, "r") as series:
        time, values = (read_column(series, key) for key in ("time", name))
    return peak(time, values)


def read_column(series: h5py.File, name: str) -> torch.Tensor:
    """Return one dataset of a time series."""
    dataset = series[name]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"time series entry {name!r} is not a dataset")
    return torch.from_numpy(dataset[()])


def sampled_diagnostics(mesh: Mesh, equation: Equation) -> dict[str, Diagnostic]:
    """Return the diagnostics written into each snapshot, by name."""
    shells = Shells(mesh)
    edges = torch.linspace(
        -DERIVATIVE_BOUND,
        DERIVATIVE_BOUND,
        DERIVATIVE_BINS + 1,
        dtype=mesh.dtype,
        device=mesh.device,
    )

    def energy_spectrum(state: State, time: float) -> torch.Tensor:
        return spectra.energy_spectrum(shells, state["velocity"])

    def budget_spectrum(term: str) -> Diagnostic:
        def budget(state: State, time: float) -> torch.Tensor:
            contribution_spectral = equation.contributions(state, time)[term]
            return spectra.budget_spectrum(
                shells, state["velocity"], contribution_spectral
            )

        return budget

    def derivative_moments(state: State, time: float) -> torch.Tensor:
        derivatives_physical = diagnostics.longitudinal_derivatives(
            mesh, state["velocity"]
        )
        return distributions.raw_moments(mesh, derivatives_physical)

    def derivative_counts(state: State, time: float) -> torch.Tensor:
        derivatives_physical = diagnostics.longitudinal_derivatives(
            mesh, state["velocity"]
        )
        return distributions.histogram(mesh, derivatives_physical, edges)

    return {
        "energy_spectrum": energy_spectrum,
        "advection_budget_spectrum": budget_spectrum("advection"),
        "diffusion_budget_spectrum": budget_spectrum("diffusion"),
        "longitudinal_derivative_moments": derivative_moments,
        "longitudinal_derivative_counts": derivative_counts,
    }


def main() -> None:
    """Run the case, or resume it, and log its peak dissipation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points", type=int, help="the number of points N")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    points: int = arguments.points
    output: Path = arguments.output or Path(f"taylor_green_3d_{points}")
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    outputs = Outputs(
        output,
        SNAPSHOT_INTERVAL,
        named,
        resume=arguments.resume,
        sampled=sampled_diagnostics(mesh, equation),
    )
    if arguments.resume:
        snapshot = outputs.latest_snapshot(mesh)
        state, start, initial = snapshot.state, snapshot.time, snapshot.time_step
    else:
        velocity_physical = flows.taylor_green_3d_physical(mesh)
        state = {"velocity": mesh.decomposition.to_spectral(velocity_physical)}
        start, initial = 0.0, INITIAL_TIME_STEP
    time_step = CFLTimeStep("velocity", CFL_TARGET, initial)
    simulation = Simulation(mesh, [equation], IFRK4(), time_step, outputs)
    simulation.run(state, start, END)

    peak_time, peak_dissipation = series_peak(output, "dissipation")
    reference_time, reference_dissipation = reference_peak()
    logger.info(
        f"peak dissipation {peak_dissipation:.5g} at t = {peak_time:.4g}; "
        f"reference {reference_dissipation:.5g} at t = {reference_time:.4g}"
    )


if __name__ == "__main__":
    main()

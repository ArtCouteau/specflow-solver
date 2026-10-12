"""Decaying HIT with a Smagorinsky LES model and on-the-fly post-processing.

Run from this folder: `python run.py`, then `python check.py`. Statistics go to output/<time>/statistics.h5, one dataset
per target under `<name>/<kind>`. Write steps are not triggered yet (clamp_dt is disabled in the solver), so only the
initial statistics at t = 0 are written.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from specflow.core.numerics.backend import log
from specflow.core.numerics.backend.backend import NumericsBackend, NumericsParameters, Precision
from specflow.core.numerics.backend.io import IOInterface
from specflow.core.numerics.backend.reduction import SerialReduction
from specflow.core.numerics.backend.transforms import SerialTransforms
from specflow.core.numerics.mesh import Mesh, MeshLayout
from specflow.core.numerics.operations import fourier
from specflow.core.numerics.operations.projection import DivergenceFreeProjector
from specflow.core.physics.dealias import FourierSmoothing, TwoThirdsDealias
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.statistics import Accumulation, Density, IntegralValue, Spectrum
from specflow.core.physics.wrappers.wrapper import Physical, Scalar, Spectral, Tensor, Vector
from specflow.processing.numerics.node import Node
from specflow.processing.physics.HIT import budget
from specflow.simulation.control import SaveControls, SimulationControls, TimeControls
from specflow.simulation.equation import NavierStokesEquation
from specflow.simulation.integration import RungeKutta4
from specflow.simulation.solver import Solver
from specflow.simulation.terms.advection import RotationFormVectorAdvectionTerm
from specflow.simulation.terms.diffusion import ConstantDiffusivityVectorDiffusionTerm
from specflow.simulation.terms.les import SmagorinskyLESModelTerm

N = 32
VISCOSITY = 0.01
PEAK_WAVENUMBER = 4.0
KINETIC_ENERGY = 0.5

TERMS = ("Advection", "Diffusion", "LES")

STATISTICS: tuple[Node[Any], ...] = (
    Node("KineticEnergy", IntegralValue[Scalar]),
    Node("ReynoldsStressTensor", IntegralValue[Tensor]),
    Node("Dissipation", IntegralValue[Scalar]),
    Node("Enstrophy", IntegralValue[Scalar]),
    Node("RmsVelocity", IntegralValue[Scalar]),
    Node("TaylorMicroscale", IntegralValue[Scalar]),
    Node("TaylorReynoldsNumber", IntegralValue[Scalar]),
    Node("KolmogorovLengthScale", IntegralValue[Scalar]),
    Node("KolmogorovTimeScale", IntegralValue[Scalar]),
    Node("LargestEddyTurnoverTime", IntegralValue[Scalar]),
    Node("ResidualDissipation", IntegralValue[Scalar]),
    Node("EffectiveKolmogorovLengthScale", IntegralValue[Scalar]),
    Node("ModalKineticEnergy", Spectrum[Scalar, Density]),
    *(Node(f"Velocity{term}EnergyTransfer", Spectrum[Scalar, Density]) for term in TERMS),
    *(Node(f"Velocity{term}EnergyTransfer", Spectrum[Scalar, Accumulation]) for term in TERMS),
)


def initial_velocity(mesh: Mesh, backend: NumericsBackend) -> Field[Vector, Spectral]:
    # White noise shaped to E(k) ~ k^4 exp(-2 (k/k0)^2), band-limited to |k| <= N/3, divergence-free
    noise = Field(
        torch.randn(
            (3, *mesh.physical.domainShape),
            dtype=backend.parameters.precision.FLOAT,
            device=backend.parameters.device,
        ),
        Vector,
        Physical,
        3,
        3,
        backend,
        mesh,
    )
    velocity = fourier.to_spectral(noise)

    k = mesh.spectral.wavenumbersSquared.data.sqrt()
    amplitude = torch.where(
        (k > 0.0) & (k <= N / 3.0),
        (k**4 * torch.exp(-2.0 * (k / PEAK_WAVENUMBER) ** 2)).sqrt() / k.clamp(min=1.0),
        0.0,
    )
    velocity = DivergenceFreeProjector().apply(mesh, velocity.replace(data=velocity.data * amplitude))

    energy = 0.5 * (mesh.spectral.parsevalWeights.data * velocity.data.abs() ** 2).sum()
    return velocity.replace(data=velocity.data * (KINETIC_ENERGY / energy).sqrt())


def main() -> None:
    torch.manual_seed(0)

    backend = NumericsBackend(
        io=IOInterface(),
        logger=log.configure("INFO"),
        parameters=NumericsParameters("cpu", Precision.from_precision("double")),
        reduction=SerialReduction(),
        transforms=SerialTransforms(),
    )
    mesh = Mesh.build(MeshLayout(N=N, LENGTH=2.0 * math.pi, DIMENSION=3), backend)

    velocity = Field(
        torch.zeros(
            (3, *mesh.spectral.domainShape),
            dtype=backend.parameters.precision.COMPLEX,
            device=backend.parameters.device,
        ),
        Vector,
        Spectral,
        3,
        3,
        backend,
        mesh,
    )

    equation = NavierStokesEquation(
        tag="Velocity",
        rank=Vector,
        terms={
            "Advection": RotationFormVectorAdvectionTerm(
                rank=Vector, tag="Advection", advecting=velocity, advected=velocity, dealias=TwoThirdsDealias(mesh)
            ),
            "Diffusion": ConstantDiffusivityVectorDiffusionTerm(
                rank=Vector, tag="Diffusion", diffused=velocity, diffusivity=VISCOSITY
            ),
            "LES": SmagorinskyLESModelTerm(
                rank=Vector,
                tag="LES",
                smoothing=FourierSmoothing(mesh),
                velocity=velocity,
                filterWidth=mesh.physical.spacing,
            ),
        },
        rhs=velocity.replace(data=torch.zeros_like(velocity.data)),
        variable=velocity,
        projector=DivergenceFreeProjector(),
    )
    equation.set_initial_conditions_spectral(initial_velocity(mesh, backend))

    solver = Solver(
        control=SimulationControls(
            time=TimeControls(current=0.0, end=0.1, save=SaveControls(every=0.05, next=0.05), step=0)
        ),
        integrator=RungeKutta4(dt=0.01, limiters={}),
        equations=[equation],
        backend=backend,
        statistics=STATISTICS,
        rules=tuple(budget.rules(equation)),
    )
    solver.run()


if __name__ == "__main__":
    main()

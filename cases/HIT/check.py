"""Checks the statistics written by run.py at the first write step. Run from this folder after run.py."""

from __future__ import annotations

import os
import re
from typing import Any

import h5py
import numpy as np
from run import STATISTICS, TERMS

from specflow.core.physics.wrappers.statistics import Accumulation, Density, IntegralValue, Spectrum
from specflow.core.physics.wrappers.wrapper import Scalar
from specflow.processing.numerics.kind import kind_tag
from specflow.processing.numerics.node import Node


def tag(node: Node[Any]) -> str:
    return f"{node.name}/{kind_tag(node.kind)}"


def main() -> None:
    first = min((d for d in os.listdir("output") if re.fullmatch(r"\d+\.\d+", d)), key=float)

    with h5py.File(f"output/{first}/statistics.h5", "r") as file:

        def read(node: Node[Any]) -> np.ndarray:
            return np.asarray(file[tag(node)])

        missing = [str(node) for node in STATISTICS if tag(node) not in file]
        assert not missing, f"Missing statistics: {missing}"

        # The initial velocity is band-limited to |k| <= N/3, so the shells hold all of its energy
        kineticEnergy = read(Node("KineticEnergy", IntegralValue[Scalar]))
        spectrum = read(Node("ModalKineticEnergy", Spectrum[Scalar, Density]))
        assert np.isclose(kineticEnergy, spectrum.sum(), rtol=1e-10), (kineticEnergy, spectrum.sum())

        for term in TERMS:
            density = read(Node(f"Velocity{term}EnergyTransfer", Spectrum[Scalar, Density]))
            accumulation = read(Node(f"Velocity{term}EnergyTransfer", Spectrum[Scalar, Accumulation]))
            assert np.isclose(accumulation[-1], density.sum(), rtol=1e-10), (term, accumulation[-1], density.sum())

        # The nonlinear term conserves energy, and viscous diffusion removes it at the dissipation rate
        advection = read(Node("VelocityAdvectionEnergyTransfer", Spectrum[Scalar, Density])).sum()
        diffusion = read(Node("VelocityDiffusionEnergyTransfer", Spectrum[Scalar, Density])).sum()
        dissipation = read(Node("Dissipation", IntegralValue[Scalar]))
        assert abs(advection) <= 1e-8 * abs(diffusion), (advection, diffusion)
        assert np.isclose(diffusion, -dissipation, rtol=1e-10), (diffusion, dissipation)

    print(f"All checks passed on output/{first}/statistics.h5")


if __name__ == "__main__":
    main()

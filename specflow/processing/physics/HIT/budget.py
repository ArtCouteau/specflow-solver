from __future__ import annotations

from typing import TYPE_CHECKING, Any

from specflow.core.physics import fields
from specflow.processing.numerics.rule import Rule, derive

if TYPE_CHECKING:
    from specflow.core.physics.wrappers.wrapper import Vector
    from specflow.simulation.equation import Equation


def rules(equation: Equation[Vector]) -> list[Rule[Any]]:
    return [
        derive(
            f"{equation.tag}{term.tag}EnergyTransfer",
            fields.spectral.velocity.modal_energy_transfer,
            equation.tag,
            f"{equation.tag}{term.tag}Contribution",
        )
        for term in equation.terms.values()
    ]

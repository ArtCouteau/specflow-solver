from __future__ import annotations

from specflow.core.physics.wrappers.constant import Constant
from specflow.core.physics.wrappers.statistics import IntegralValue
from specflow.core.physics.wrappers.wrapper import Scalar
from specflow.processing.numerics.rule import derive, register
from specflow.processing.physics.HIT.integral import if_dissipative


def effective_Kolmogorov_length_scale(
    viscosity: Constant[Scalar], eddyViscosity: IntegralValue[Scalar], dissipation: IntegralValue[Scalar]
) -> IntegralValue[Scalar]:
    return if_dissipative(dissipation, ((viscosity.value + eddyViscosity.data) ** 3 / dissipation.data).pow(0.25))


register(
    derive(
        "EffectiveKolmogorovLengthScale",
        effective_Kolmogorov_length_scale,
        "ViscosityConstant",
        "EddyViscosity",
        "Dissipation",
    ),
)

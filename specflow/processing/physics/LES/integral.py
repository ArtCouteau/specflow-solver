from __future__ import annotations

from typing import Self

from specflow.core.physics import fields
from specflow.core.physics.wrappers.wrapper import Scalar
from specflow.processing.physics.HIT.integral import DissipationRate, StrainRateTensor, if_dissipative
from specflow.processing.physics.statistics.frame import Frame
from specflow.processing.physics.statistics.inputs import SpectralEddyViscosity, ViscosityConstant
from specflow.processing.physics.statistics.quantity import Derived
from specflow.processing.physics.statistics.wrappers import IntegralValue


class ResidualDissipationRate(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        S = frame[StrainRateTensor].value
        nuT = frame[SpectralEddyViscosity].value
        return self.__class__(S.backend.reduction.global_mean(fields.physical.les.residual_dissipation(S, nuT)))


class EffectiveKolmogorovLengthScale(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        epsilon = frame[DissipationRate].value
        nu = frame[ViscosityConstant].value
        nuT = frame[SpectralEddyViscosity].value
        return self.__class__(if_dissipative(epsilon, ((nu.value + nuT.data[0, 0, 0]) ** 3 / epsilon.data).pow(0.25)))


ALL = (ResidualDissipationRate,)

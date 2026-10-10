from __future__ import annotations

from typing import Self

from specflow.core.numerics.operations import fourier
from specflow.core.physics import fields
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, Scalar
from specflow.processing.physics.HIT.integral import DissipationRate, StrainRateTensor, if_dissipative
from specflow.processing.physics.statistics.frame import Frame
from specflow.processing.physics.statistics.inputs import SpectralEddyViscosity, ViscosityConstant
from specflow.processing.physics.statistics.quantity import Derived
from specflow.processing.physics.statistics.wrappers import IntegralValue


class PhysicalEddyViscosity(Derived[Field[Scalar, Physical]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(fourier.to_physical(frame[SpectralEddyViscosity]))


class ResidualDissipationRate(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(
            frame.backend.reduction.global_mean(
                fields.physical.les.residual_dissipation(frame[StrainRateTensor], frame[PhysicalEddyViscosity])
            )
        )


class EffectiveKolmogorovLengthScale(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(
            if_dissipative(
                frame[DissipationRate],
                (
                    (frame[ViscosityConstant].value + frame[SpectralEddyViscosity].data[0, 0, 0]) ** 3
                    / frame[DissipationRate].data
                ).pow(0.25),
            )
        )


ALL = (ResidualDissipationRate,)

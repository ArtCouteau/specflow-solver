from __future__ import annotations

from typing import Self

import torch

from specflow.core.physics import fields
from specflow.core.physics.wrappers.wrapper import Scalar, Tensor
from specflow.processing.physics.HIT.fields import PhysicalVelocity, RotationRateTensor, StrainRateTensor
from specflow.processing.physics.statistics.frame import Frame
from specflow.processing.physics.statistics.inputs import ViscosityConstant
from specflow.processing.physics.statistics.quantity import Derived
from specflow.processing.physics.statistics.wrappers import IntegralValue


def if_dissipative(epsilon: IntegralValue[Scalar], value: torch.Tensor) -> IntegralValue[Scalar]:
    return IntegralValue(torch.where(epsilon.data > 0.0, value, torch.inf), Scalar)


class TotalKineticEnergy(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(
            frame.backend.reduction.global_mean(fields.physical.velocity.kinetic_energy(frame[PhysicalVelocity]))
        )


class ReynoldsStressTensor(Derived[IntegralValue[Tensor]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(
            frame.backend.reduction.global_mean(
                fields.physical.velocity.Reynolds_stress_tensor(frame[PhysicalVelocity])
            )
        )


class RmsVelocity(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(IntegralValue(frame[TotalKineticEnergy].data.sqrt(), Scalar))


class DissipationRate(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(
            frame.backend.reduction.global_mean(
                fields.physical.velocity.dissipation(frame[StrainRateTensor], frame[ViscosityConstant])
            )
        )


class TotalEnstrophy(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(
            frame.backend.reduction.global_mean(fields.physical.velocity.enstrophy(frame[RotationRateTensor]))
        )


class TaylorMicroscale(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(
            if_dissipative(
                frame[DissipationRate],
                frame[RmsVelocity].data * (15.0 * frame[ViscosityConstant].value / frame[DissipationRate].data).sqrt(),
            )
        )


class TaylorReynoldsNumber(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(
            IntegralValue(
                frame[RmsVelocity].data * frame[TaylorMicroscale].data / frame[ViscosityConstant].value, Scalar
            )
        )


class KolmogorovLengthScale(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(
            if_dissipative(
                frame[DissipationRate],
                (frame[ViscosityConstant].value ** 3 / frame[DissipationRate].data).pow(0.25),
            )
        )


class KolmogorovTimeScale(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(
            if_dissipative(
                frame[DissipationRate],
                (frame[ViscosityConstant].value / frame[DissipationRate].data).pow(0.5),
            )
        )


class LargestEddyTurnoverTime(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(
            if_dissipative(frame[DissipationRate], frame[TotalKineticEnergy].data / frame[DissipationRate].data)
        )


SCALES = (TaylorMicroscale, TaylorReynoldsNumber, KolmogorovLengthScale, KolmogorovTimeScale, LargestEddyTurnoverTime)
ALL = (TotalKineticEnergy, ReynoldsStressTensor, RmsVelocity, DissipationRate, TotalEnstrophy, *SCALES)

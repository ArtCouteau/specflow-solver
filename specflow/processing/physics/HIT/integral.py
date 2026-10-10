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
        u = frame[PhysicalVelocity]
        return self.__class__(u.backend.reduction.global_mean(fields.physical.velocity.kinetic_energy(u)))


class ReynoldsStressTensor(Derived[IntegralValue[Tensor]]):
    def compute(self, frame: Frame) -> Self:
        u = frame[PhysicalVelocity]
        return self.__class__(u.backend.reduction.global_mean(fields.physical.velocity.Reynolds_stress_tensor(u)))


class RmsVelocity(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(IntegralValue(frame[TotalKineticEnergy].data.sqrt(), Scalar))


class DissipationRate(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        S = frame[StrainRateTensor]
        nu = frame[ViscosityConstant]
        return self.__class__(S.backend.reduction.global_mean(fields.physical.velocity.dissipation(S, nu)))


class TotalEnstrophy(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        Omega = frame[RotationRateTensor]
        return self.__class__(Omega.backend.reduction.global_mean(fields.physical.velocity.enstrophy(Omega)))


class TaylorMicroscale(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        u = frame[RmsVelocity]
        epsilon = frame[DissipationRate]
        nu = frame[ViscosityConstant]
        return self.__class__(if_dissipative(epsilon, u.data * (15.0 * nu.value / epsilon.data).sqrt()))


class TaylorReynoldsNumber(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        u = frame[RmsVelocity]
        lambda_ = frame[TaylorMicroscale]
        nu = frame[ViscosityConstant]
        return self.__class__(IntegralValue(u.data * lambda_.data / nu.value, Scalar))


class KolmogorovLengthScale(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        epsilon = frame[DissipationRate]
        nu = frame[ViscosityConstant]
        return self.__class__(if_dissipative(epsilon, (nu.value**3 / epsilon.data).pow(0.25)))


class KolmogorovTimeScale(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        epsilon = frame[DissipationRate]
        nu = frame[ViscosityConstant]
        return self.__class__(if_dissipative(epsilon, (nu.value / epsilon.data).pow(0.5)))


class LargestEddyTurnoverTime(Derived[IntegralValue[Scalar]]):
    def compute(self, frame: Frame) -> Self:
        k = frame[TotalKineticEnergy]
        epsilon = frame[DissipationRate]
        return self.__class__(if_dissipative(epsilon, k.data / epsilon.data))


SCALES = (TaylorMicroscale, TaylorReynoldsNumber, KolmogorovLengthScale, KolmogorovTimeScale, LargestEddyTurnoverTime)
ALL = (TotalKineticEnergy, ReynoldsStressTensor, RmsVelocity, DissipationRate, TotalEnstrophy, *SCALES)

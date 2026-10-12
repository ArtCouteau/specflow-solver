from __future__ import annotations

import torch

from specflow.core.physics.wrappers.constant import Constant
from specflow.core.physics.wrappers.statistics import IntegralValue
from specflow.core.physics.wrappers.wrapper import Scalar
from specflow.processing.numerics.rule import derive, register


def if_dissipative(dissipation: IntegralValue[Scalar], value: torch.Tensor) -> IntegralValue[Scalar]:
    return IntegralValue(torch.where(dissipation.data > 0.0, value, torch.inf), Scalar)


def rms_velocity(kineticEnergy: IntegralValue[Scalar]) -> IntegralValue[Scalar]:
    return IntegralValue(kineticEnergy.data.sqrt(), Scalar)


def Taylor_microscale(
    rmsVelocity: IntegralValue[Scalar], viscosity: Constant[Scalar], dissipation: IntegralValue[Scalar]
) -> IntegralValue[Scalar]:
    return if_dissipative(dissipation, rmsVelocity.data * (15.0 * viscosity.value / dissipation.data).sqrt())


def Taylor_Reynolds_number(
    rmsVelocity: IntegralValue[Scalar], TaylorMicroscale: IntegralValue[Scalar], viscosity: Constant[Scalar]
) -> IntegralValue[Scalar]:
    return IntegralValue(rmsVelocity.data * TaylorMicroscale.data / viscosity.value, Scalar)


def Kolmogorov_length_scale(viscosity: Constant[Scalar], dissipation: IntegralValue[Scalar]) -> IntegralValue[Scalar]:
    return if_dissipative(dissipation, (viscosity.value**3 / dissipation.data).pow(0.25))


def Kolmogorov_time_scale(viscosity: Constant[Scalar], dissipation: IntegralValue[Scalar]) -> IntegralValue[Scalar]:
    return if_dissipative(dissipation, (viscosity.value / dissipation.data).pow(0.5))


def largest_eddy_turnover_time(
    kineticEnergy: IntegralValue[Scalar], dissipation: IntegralValue[Scalar]
) -> IntegralValue[Scalar]:
    return if_dissipative(dissipation, kineticEnergy.data / dissipation.data)


register(
    derive("RmsVelocity", rms_velocity, "KineticEnergy"),
    derive("TaylorMicroscale", Taylor_microscale, "RmsVelocity", "ViscosityConstant", "Dissipation"),
    derive("TaylorReynoldsNumber", Taylor_Reynolds_number, "RmsVelocity", "TaylorMicroscale", "ViscosityConstant"),
    derive("KolmogorovLengthScale", Kolmogorov_length_scale, "ViscosityConstant", "Dissipation"),
    derive("KolmogorovTimeScale", Kolmogorov_time_scale, "ViscosityConstant", "Dissipation"),
    derive("LargestEddyTurnoverTime", largest_eddy_turnover_time, "KineticEnergy", "Dissipation"),
)

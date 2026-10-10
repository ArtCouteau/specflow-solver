from __future__ import annotations

from typing import Self

from specflow.core.physics import fields
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Physical, Tensor, Vector
from specflow.processing.physics.statistics.frame import Frame
from specflow.processing.physics.statistics.inputs import SpectralVelocity
from specflow.processing.physics.statistics.quantity import Derived


class PhysicalVelocity(Derived[Field[Vector, Physical]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(fields.physical.velocity.physical_velocity(frame[SpectralVelocity].value))


class VelocityGradient(Derived[Field[Tensor, Physical]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(fields.physical.velocity.velocity_gradient(frame[SpectralVelocity].value))


class StrainRateTensor(Derived[Field[Tensor, Physical]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(fields.physical.velocity.strain_rate_tensor(frame[VelocityGradient].value))


class RotationRateTensor(Derived[Field[Tensor, Physical]]):
    def compute(self, frame: Frame) -> Self:
        return self.__class__(fields.physical.velocity.rotation_rate_tensor(frame[VelocityGradient].value))

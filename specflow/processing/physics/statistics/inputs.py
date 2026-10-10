from __future__ import annotations

from specflow.core.physics.wrappers.constant import Constant
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Scalar, Spectral, Vector
from specflow.processing.physics.statistics.quantity import Input


class SpectralVelocity(Input[Field[Vector, Spectral]]): ...


class SpectralEddyViscosity(Input[Field[Scalar, Spectral]]): ...


class ViscosityConstant(Input[Constant[Scalar]]): ...

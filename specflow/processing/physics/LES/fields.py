from specflow.core.physics import fields
from specflow.processing.numerics.rule import derive, register

register(
    derive("ResidualDissipation", fields.physical.les.residual_dissipation, "StrainRateTensor", "EddyViscosity"),
)

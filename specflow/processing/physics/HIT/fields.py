from specflow.core.physics import fields
from specflow.processing.numerics.rule import derive, register

register(
    derive("VelocityGradient", fields.physical.velocity.velocity_gradient, "Velocity"),
    derive("StrainRateTensor", fields.physical.velocity.strain_rate_tensor, "VelocityGradient"),
    derive("RotationRateTensor", fields.physical.velocity.rotation_rate_tensor, "VelocityGradient"),
    derive("KineticEnergy", fields.physical.velocity.kinetic_energy, "Velocity"),
    derive("ReynoldsStressTensor", fields.physical.velocity.Reynolds_stress_tensor, "Velocity"),
    derive("Dissipation", fields.physical.velocity.dissipation, "StrainRateTensor", "ViscosityConstant"),
    derive("Enstrophy", fields.physical.velocity.enstrophy, "RotationRateTensor"),
)

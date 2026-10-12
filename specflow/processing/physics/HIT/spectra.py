from specflow.core.physics import fields
from specflow.processing.numerics.rule import derive, register

register(derive("ModalKineticEnergy", fields.spectral.velocity.modal_kinetic_energy, "Velocity"))

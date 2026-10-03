from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Physical, Spectral, TRank


def to_spectral(field: Field[TRank, Physical]) -> Field[TRank, Spectral]:
    return field.backend.transforms.rfftn(field)


def to_physical(field: Field[TRank, Spectral]) -> Field[TRank, Physical]:
    return field.backend.transforms.irfftn(field)

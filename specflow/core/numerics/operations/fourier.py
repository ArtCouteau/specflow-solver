from specflow.core.numerics.backend.transforms import TransformsBackend
from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Physical, Spectral, TRank


def to_spectral(transforms: TransformsBackend, field: Field[TRank, Physical]) -> Field[TRank, Spectral]:
    return transforms.rfftn(field)


def to_physical(transforms: TransformsBackend, field: Field[TRank, Spectral]) -> Field[TRank, Physical]:
    return transforms.irfftn(field)

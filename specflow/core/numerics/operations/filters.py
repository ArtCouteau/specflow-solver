from specflow.core.numerics.mesh import Mesh
from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Scalar, Spectral


def spherical_truncation(mesh: Mesh, maxWavenumber: float) -> Field[Scalar, Spectral]:
    return mesh.spectral.wavenumbersSquared.replace(data=mesh.spectral.wavenumbersSquared.data < (maxWavenumber**2))

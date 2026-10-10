from specflow.core.numerics.mesh import Mesh
from specflow.core.physics.wrappers.field import Field
from specflow.core.physics.wrappers.wrapper import Scalar, Spectral


def spherical_truncation(mesh: Mesh, maxWavenumber: float) -> Field[Scalar, Spectral]:
    return mesh.spectral.wavenumbersSquared.replace(data=mesh.spectral.wavenumbersSquared.data < (maxWavenumber**2))


def spherical_shell(mesh: Mesh, lower: float, upper: float) -> Field[Scalar, Spectral]:
    return mesh.spectral.wavenumbersSquared.replace(
        data=(mesh.spectral.wavenumbersSquared.data > (lower**2)) & (mesh.spectral.wavenumbersSquared.data < (upper**2))
    )

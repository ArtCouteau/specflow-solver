import jax.numpy as jnp
from jax import Array
from jaxtyping import Inexact

from specflow.core.numerics.mesh import Mesh

def transpose(tensor: Inexact[Array, "C C, *domain"]) -> Inexact[Array, "C C, *domain"]:
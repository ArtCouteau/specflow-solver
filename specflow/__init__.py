__author__ = 'Arthur Couteau <acouteau@ethz.ch>'
__license__ = 'GNU Lesser GPL version 3 or any later version'

from .core import init
from .core.numerics import get_physical_grid, get_spectral_grid
from . import simulation

__all__ = ['init', 'simulation', 'get_physical_grid', 'get_spectral_grid']

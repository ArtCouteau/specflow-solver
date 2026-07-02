
import logging

from .. import context


def configure() -> logging.Logger:
    logging.basicConfig(level=context.get().config.get('numerics/logging', str, 'INFO').upper())

    return logging.getLogger('SPECFLOW')

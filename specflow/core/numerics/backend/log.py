import logging
from typing import Literal


def configure(level: Literal["DEBUG", "WARNING", "INFO"] = "INFO") -> logging.Logger:
    logging.basicConfig(level=level)

    return logging.getLogger("SPECFLOW")

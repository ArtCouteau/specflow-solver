
import logging


def info(message: str, level: int = 0) -> None:
    logging.info('  '*level + message)


def debug(message: str) -> None:
    logging.debug(message)


def warning(message: str) -> None:
    logging.warning(message)

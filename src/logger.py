import sys

from loguru import logger  # type: ignore


def setup_logger() -> None:
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | <cyan>{name}</cyan>:"
        "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )


def get_logger(name: str = __name__) -> logger:
    return logger.bind(name=name)

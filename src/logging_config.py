# src/logging_config.py
import logging
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for the package.

    Args:
        level: Logging level (e.g., logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
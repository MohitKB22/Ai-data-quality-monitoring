"""Standardized logger factory used across the project."""

import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

_CONFIGURED_LOGGERS = set()


def get_logger(name: str, log_file: str = "pipeline.log", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    os.makedirs(LOGS_DIR, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(LOGS_DIR, log_file))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED_LOGGERS.add(name)
    return logger

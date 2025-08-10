# -*- coding: utf-8 -*-
import logging, os
from logging.handlers import RotatingFileHandler

_LOGGER = None

def setup_logging(log_path: str):
    global _LOGGER
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("snaplite")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt); ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _LOGGER = logger
    logger.info("logging initialized: %s", log_path)

def get_logger():
    return _LOGGER or logging.getLogger("snaplite")

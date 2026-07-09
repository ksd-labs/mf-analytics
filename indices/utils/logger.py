"""
utils/logger.py
"""

import logging

from pathlib import Path

from indices.config.index_metadata import (
    LOG_DIR,
)

def get_logger(
    name: str
):

    logger = logging.getLogger(
        name
    )

    if logger.handlers:
        return logger

    logger.setLevel(
        logging.INFO
    )

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )

    log_file = (
        Path(LOG_DIR)
        / "nse_downloader.log"
    )

    file_handler = (
        logging.FileHandler(
            log_file
        )
    )

    file_handler.setFormatter(
        formatter
    )

    console_handler = (
        logging.StreamHandler()
    )

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console_handler
    )

    return logger
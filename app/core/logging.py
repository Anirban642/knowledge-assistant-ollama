"""Centralised logging configuration for the application."""

import logging
import logging.config


def configure_logging(log_level: str = "INFO") -> None:
    """Configure the root logger to print readable lines to the console."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {"level": log_level.upper(), "handlers": ["console"]},
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module. Usage: logger = get_logger(__name__)."""
    return logging.getLogger(name)
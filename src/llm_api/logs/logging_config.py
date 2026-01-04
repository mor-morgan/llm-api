"""
Central logging configuration for the project.

Responsibilities:
- Configure application-wide logging in a single place
- Provide a consistent log format across all modules
- Avoid duplicated log entries (important when using uvicorn --reload)
- Support environment-based log levels (via LOG_LEVEL)

Notes:
- Console logging is used (suitable for local development and Docker)
- request_id support can be added later using logging.LoggerAdapter
"""

import logging
import os
import sys


def setup_logging() -> None:
    """
    Initialize root logging configuration.

    This function should be called exactly once during application startup
    (e.g. in app/main.py), before any loggers are used.
    """

    # Resolve log level from environment variable
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()

    # Prevent duplicate logs if logging is configured multiple times
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
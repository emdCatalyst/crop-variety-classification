"""Loguru-based logging setup.

Routes Python's stdlib logging records (used by uvicorn, fastapi, sqlalchemy)
into loguru so a single sink renders everything. JSON in non-debug mode for
log shippers; pretty in debug mode for local dev.
"""
from __future__ import annotations

import logging
import sys

from loguru import logger

_INTERCEPT_LOGGERS = (
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "fastapi",
    "sqlalchemy",
    "sqlalchemy.engine",
    "agrovision",
    "agrovision.inference",
)


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__ and frame.f_back:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


_configured = False


def setup_logging(debug: bool = False, pretty: bool = True) -> None:
    global _configured
    if _configured:
        return
    _configured = True

    logger.remove()
    logger.add(
        sys.stdout,
        level="DEBUG" if debug else "INFO",
        serialize=not pretty,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )

    handler = _InterceptHandler()
    logging.basicConfig(handlers=[handler], level=logging.INFO, force=True)
    for name in _INTERCEPT_LOGGERS:
        log = logging.getLogger(name)
        log.handlers = [handler]
        log.propagate = False

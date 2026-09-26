"""Structured JSON logging (structlog), one run_id per pipeline execution."""

from __future__ import annotations

import logging
import sys
import uuid

import structlog
from structlog.typing import FilteringBoundLogger

from predictor.config import get_settings


def configure_logging(level: str | None = None) -> None:
    """Configure structlog to emit one JSON object per line on stdout."""
    resolved = (level or get_settings().app.log_level).upper()
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=resolved, force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(resolved)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def bind_run_id(run_id: str | None = None) -> str:
    """Bind a run_id to the logging context for the whole execution and return it."""
    resolved = run_id or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(run_id=resolved)
    return resolved


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    logger: FilteringBoundLogger = structlog.get_logger(name)
    return logger

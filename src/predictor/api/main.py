"""FastAPI application. Phase 0 exposes liveness only; real endpoints arrive in phase 7."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Literal

from fastapi import FastAPI, Response
from pydantic import BaseModel

from predictor.config import Settings, get_settings
from predictor.db import check_connection
from predictor.logging import bind_run_id, configure_logging, get_logger

logger = get_logger(__name__)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    db: Literal["ok", "error"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Loading the settings here means an invalid config fails the container, not a request.
    settings: Settings = get_settings()
    configure_logging(settings.app.log_level)
    bind_run_id()
    logger.info(
        "api.startup",
        competitions=list(settings.competition_codes),
        version=settings.app.version,
    )
    yield
    logger.info("api.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    return FastAPI(
        title="Sports Results Prediction Platform",
        version=settings.app.version,
        lifespan=lifespan,
    )


app = create_app()


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """Liveness plus database connectivity."""
    db_ok = check_connection()
    if not db_ok:
        response.status_code = HTTPStatus.SERVICE_UNAVAILABLE
        logger.warning("health.db_unreachable")
    return HealthResponse(status="ok" if db_ok else "degraded", db="ok" if db_ok else "error")

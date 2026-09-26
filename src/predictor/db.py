"""Database access: SQLAlchemy 2 engine, sessions and a connectivity probe."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import URL, Engine, create_engine, make_url, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from predictor.config import Settings, get_settings

#: Full connection string override (used by CI and by tooling outside the compose network).
DATABASE_URL_ENV = "DATABASE_URL"

DRIVER = "postgresql+psycopg"


class Base(DeclarativeBase):
    """Declarative base. Tables arrive with their Alembic migrations, phase by phase."""


def build_url(settings: Settings | None = None) -> URL:
    """Build the connection URL from DATABASE_URL, or from the settings plus the env password."""
    override = os.environ.get(DATABASE_URL_ENV)
    if override:
        return make_url(override).set(drivername=DRIVER)
    resolved = settings or get_settings()
    return URL.create(
        drivername=DRIVER,
        username=resolved.database.user,
        password=resolved.database.password.get_secret_value() or None,
        host=resolved.database.host,
        port=resolved.database.port,
        database=resolved.database.name,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and cache the engine. Call `get_engine.cache_clear()` in tests."""
    settings = get_settings()
    return create_engine(
        build_url(settings),
        pool_pre_ping=True,
        future=True,
        # Without this, an unreachable host blocks on DNS/TCP for the OS default.
        connect_args={"connect_timeout": settings.database.connect_timeout},
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session: commit on success, roll back on failure."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Return whether the database answers `SELECT 1`."""
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return False
    return True

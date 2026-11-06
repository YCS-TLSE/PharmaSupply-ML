"""
===============================================================================
PharmaSupply-ML: Database Engine & Cursor Helpers
===============================================================================

PURPOSE
-------
Provide a single pooled SQLAlchemy engine and a safe cursor context manager
for every PostgreSQL access in the project.

RESPONSIBILITIES
----------------
* Lazily create the engine with ``pool_pre_ping`` and bounded pool settings.
* Borrow a DBAPI connection from the pool and always return it.
* Convert driver/SQLAlchemy failures into ``DatabaseUnavailableError`` so the
  HTTP layer can return an explicit ``503`` instead of fabricated data.
* Offer a cheap ``SELECT 1`` health check for readiness probes.

INPUTS / OUTPUTS
----------------
Inputs: ``DATABASE_URL`` or ``POSTGRES_*`` environment variables.
Outputs: SQLAlchemy ``Engine`` instances, DBAPI cursors, and a boolean health
signal. No business data is interpreted here.

ARCHITECTURE INTEGRATION
------------------------
Used by ``src.repositories.inventory`` (serving), ``src.ingestion`` and
``src.validation``. The engine is process-wide; tests can call
``reset_engine()`` to isolate connection state.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config import (
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT,
)
from src.db_config import get_database_url

logger = logging.getLogger("pharmasupply.repository")

# Process-wide engine. ``None`` means "not created yet"; ``get_engine`` is the
# only supported way to obtain it.
_engine: Engine | None = None


# Domain-specific exception raised when PostgreSQL cannot serve a request.
# Routers catch this type and translate it to HTTP 503.
class DatabaseUnavailableError(RuntimeError):
    """Raised when PostgreSQL cannot serve a request."""


# Lazy singleton. ``pool_pre_ping`` avoids handing out dead connections after
# a database restart; pool parameters are configurable through src.config.
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine (lazy singleton)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_timeout=DB_POOL_TIMEOUT,
        )
    return _engine


# Test/utility helper: dispose the pool and forget the engine so the next call
# rebuilds it from the current environment.
def reset_engine() -> None:
    """Dispose and forget the current engine (useful for tests)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def get_db_connection():
    """Borrow a pooled DBAPI connection from the SQLAlchemy engine."""
    return get_engine().raw_connection()


# Context manager used by every repository query. It yields a cursor, closes
# the cursor, and returns the connection to the pool in all cases.
@contextmanager
def db_cursor(dict_rows: bool = False) -> Iterator[object]:
    """Yield a cursor and always return its connection to the pool."""
    try:
        connection = get_db_connection()
    except (
        psycopg2.Error,
        SQLAlchemyError,
        RuntimeError,
    ) as exc:
        raise DatabaseUnavailableError(str(exc)) from exc

    try:
        factory = (
            psycopg2.extras.DictCursor
            if dict_rows
            else None
        )
        with connection.cursor(
            cursor_factory=factory
        ) as cursor:
            yield cursor
    except (psycopg2.Error, SQLAlchemyError) as exc:
        raise DatabaseUnavailableError(str(exc)) from exc
    finally:
        connection.close()


# Readiness helper: a successful trivial query means the pool and database are
# usable. Any failure is reported as ``False`` rather than raising.
def check_database_health() -> bool:
    """Return True when a trivial query succeeds."""
    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT 1;")
            cursor.fetchone()
        return True
    except DatabaseUnavailableError:
        return False

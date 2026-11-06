"""
Database helper tests.

These tests exercise the engine singleton, cursor lifecycle, connection
failure translation and readiness probe without requiring a live PostgreSQL
instance. Driver calls are stubbed so the behaviour is deterministic.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.repositories.database import (
    DatabaseUnavailableError,
    check_database_health,
    db_cursor,
    reset_engine,
)


class _HealthyCursor:
    def execute(self, query):
        self.query = query

    def fetchone(self):
        return (1,)


@contextmanager
def _healthy_cursor(dict_rows=False):
    yield _HealthyCursor()


def test_check_database_health_true():
    with patch(
        "src.repositories.database.db_cursor",
        _healthy_cursor,
    ):
        assert check_database_health() is True


def test_check_database_health_false():
    with patch(
        "src.repositories.database.db_cursor",
        side_effect=DatabaseUnavailableError("down"),
    ):
        assert check_database_health() is False


def test_db_cursor_wraps_connection_failure():
    with patch(
        "src.repositories.database.get_db_connection",
        side_effect=RuntimeError("no dsn"),
    ):
        with pytest.raises(DatabaseUnavailableError):
            with db_cursor():
                pass


def test_reset_engine_is_idempotent():
    reset_engine()
    reset_engine()


class _CursorContext:
    def __enter__(self):
        return MagicMock()

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self):
        self.closed = False

    def cursor(self, cursor_factory=None):
        return _CursorContext()

    def close(self):
        self.closed = True


def test_db_cursor_closes_connection():
    connection = _FakeConnection()
    with patch(
        "src.repositories.database.get_db_connection",
        return_value=connection,
    ):
        with db_cursor() as cursor:
            cursor.execute("SELECT 1;")
    assert connection.closed is True


def test_get_engine_is_a_singleton(monkeypatch):
    import src.repositories.database as database

    database.reset_engine()
    fake_engine = MagicMock()
    monkeypatch.setattr(
        database,
        "get_database_url",
        lambda: "postgresql://fake/db",
    )
    monkeypatch.setattr(
        database,
        "create_engine",
        lambda *a, **k: fake_engine,
    )
    assert database.get_engine() is fake_engine
    assert database.get_engine() is fake_engine
    database.reset_engine()
    fake_engine.dispose.assert_called_once()

"""
Database URL resolution tests.

Ensures an explicit ``DATABASE_URL`` wins, component-based URLs are built
correctly, and missing secrets raise an actionable error instead of silently
falling back to an insecure default.
"""

from __future__ import annotations

import pytest

from src.db_config import get_database_url


def test_explicit_database_url_wins(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://explicit/db"
    )
    assert get_database_url() == "postgresql://explicit/db"


def test_database_url_from_components(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_USER", "user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_DB", "pharma")
    monkeypatch.setenv("POSTGRES_HOST", "db")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    assert (
        get_database_url()
        == "postgresql://user:secret@db:5432/pharma"
    )


def test_missing_secrets_raise(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name in (
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError):
        get_database_url()

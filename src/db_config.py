"""Database settings from the environment. Secrets have no in-code fallbacks."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

_REQUIRED = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")


def get_database_url() -> str:
    """Return DATABASE_URL, or build it from POSTGRES_* vars.

    Host and port may default (not secrets). User, password, and database
    name must be set. Raises at connection time, not at import, so tests
    can load the API module without a live database.
    """
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit

    missing = [name for name in _REQUIRED if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required database environment variable(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and set the values."
        )

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    name = os.environ["POSTGRES_DB"]
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"

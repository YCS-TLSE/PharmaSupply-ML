"""
Model status and notification tests.

Validates the fail-safe status parser (including ``UNKNOWN``), the no-op
behaviour when SMTP credentials are missing, and the cancellable transition
watcher used by the API lifespan.
"""

from __future__ import annotations

import asyncio

import pytest

import src.services.model_status as model_status


def test_read_missing_status_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_status,
        "STATUS_FILE_PATH",
        str(tmp_path / "missing.txt"),
    )
    assert model_status.read_model_status() == "UNKNOWN"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("OK", "STABLE"),
        ("STABLE", "STABLE"),
        ("WARNING", "WARNING"),
        ("CRITICAL", "CRITICAL"),
        ("garbage", "UNKNOWN"),
    ],
)
def test_read_status_values(
    tmp_path, monkeypatch, content, expected
):
    path = tmp_path / "status.txt"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        model_status, "STATUS_FILE_PATH", str(path)
    )
    assert model_status.read_model_status() == expected


def test_send_email_without_password_is_noop(monkeypatch):
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    model_status.send_support_email("CRITICAL")


def test_watcher_is_cancellable(monkeypatch):
    monkeypatch.setattr(
        model_status, "MODEL_STATUS_POLL_SECONDS", 0.01
    )
    monkeypatch.setattr(
        model_status, "read_model_status", lambda: "STABLE"
    )

    async def _run():
        task = asyncio.create_task(
            model_status.watch_model_status()
        )
        await asyncio.sleep(0.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())

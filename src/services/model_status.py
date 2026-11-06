"""
===============================================================================
PharmaSupply-ML: Model Status & Drift Notifications
===============================================================================

PURPOSE
-------
Read the drift status produced by the monitoring batch and dispatch e-mail
notifications without putting SMTP latency on the HTTP request path.

RESPONSIBILITIES
----------------
* Parse ``model_status.txt`` into ``STABLE``/``WARNING``/``CRITICAL`` or the
  fail-safe ``UNKNOWN`` state.
* Send a tailored e-mail on WARNING/CRITICAL transitions only.
* Run a cancellable asyncio watcher started by the API lifespan.

INPUTS / OUTPUTS
----------------
Inputs: the shared status file and SMTP environment variables.
Outputs: a status string, log records and best-effort e-mail messages.

ARCHITECTURE INTEGRATION
------------------------
``src.api`` starts ``watch_model_status`` in its lifespan. Routers call
``read_model_status`` for response enrichment. The monitoring batch writes the
file atomically to avoid partial reads.

OPERATIONAL NOTES
-----------------
Missing SMTP credentials disable sending but never crash the service. The
watcher is transition-based, so a stable WARNING does not flood the mailbox.
"""

from __future__ import annotations

import asyncio
import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from src.config import (
    MODEL_STATUS_POLL_SECONDS,
    MODEL_STATUS_VALUES,
    RECEIVER_EMAIL_DEFAULT,
    SENDER_EMAIL_DEFAULT,
    SMTP_PORT_DEFAULT,
    SMTP_SERVER_DEFAULT,
    SMTP_TIMEOUT_SECONDS,
    STATUS_FILE_PATH,
)

logger = logging.getLogger("pharmasupply.model_status")


# Notification is best-effort by design: SMTP outages must not affect the API.
# The function is called from a worker thread by the async watcher.
def send_support_email(drift_status: str) -> None:
    """Send a drift notification e-mail (called from the background watcher only)."""
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    if not smtp_password:
        logger.warning(
            "Missing SMTP password. Email dispatch skipped."
        )
        return

    msg = EmailMessage()
    if drift_status == "CRITICAL":
        msg["Subject"] = (
            "🚨 [ALERTE CRITIQUE] Instabilité majeure du modèle de prévision"
        )
        msg.set_content(
            "Bonjour l'équipe support,\n\n"
            "Le modèle de calcul PharmaSupply a atteint un seuil de dérive CRITIQUE.\n"
            "Une action corrective ou un réentraînement est requis sous 24h.\n"
        )
    elif drift_status == "WARNING":
        msg["Subject"] = (
            "⚠️ [Alerte Modérée] Légère baisse de performance du modèle détectée"
        )
        msg.set_content(
            "Bonjour l'équipe technique,\n\n"
            "Le modèle présente une légère dérive (Statut: WARNING).\n"
            "Une vigilance est requise et nos équipes travaillent à son optimisation.\n"
        )
    else:
        return

    msg["From"] = SENDER_EMAIL_DEFAULT
    msg["To"] = RECEIVER_EMAIL_DEFAULT

    try:
        with smtplib.SMTP(
            SMTP_SERVER_DEFAULT,
            SMTP_PORT_DEFAULT,
            timeout=SMTP_TIMEOUT_SECONDS,
        ) as server:
            server.starttls()
            server.login(
                SENDER_EMAIL_DEFAULT, smtp_password
            )
            server.send_message(msg)
        logger.info(
            "Support notification email ('%s') sent.",
            drift_status,
        )
    except Exception as exc:  # noqa: BLE001 - notification must never crash
        logger.error(
            "Failed to dispatch support email: %s", exc
        )


# Side-effect-free reader used by /model/health, /alerts and /predict. Any
# unreadable or unexpected content is mapped to UNKNOWN, never to STABLE.
def read_model_status() -> str:
    """Read the model health flag without any side effect."""
    status_path = Path(STATUS_FILE_PATH)
    if not status_path.exists():
        logger.warning(
            "Model status file not found at '%s'. Reporting UNKNOWN.",
            STATUS_FILE_PATH,
        )
        return "UNKNOWN"

    try:
        content = status_path.read_text(
            encoding="utf-8"
        ).strip()
    except OSError as exc:
        logger.warning(
            "Unable to read model status file '%s': %s. Reporting UNKNOWN.",
            STATUS_FILE_PATH,
            exc,
        )
        return "UNKNOWN"

    if content == "OK":
        return "STABLE"
    if content in MODEL_STATUS_VALUES:
        return content

    logger.warning(
        "Unexpected model status content %r. Reporting UNKNOWN.",
        content,
    )
    return "UNKNOWN"


# Backward-compatible alias kept for callers that imported the old name.
def get_model_status() -> str:
    """Backward-compatible alias for :func:`read_model_status`."""
    return read_model_status()


# Async watcher started by the FastAPI lifespan. It polls the status file in a
# worker thread, alerts only on WARNING/CRITICAL transitions, and resets its
# memory when the model recovers so a future drift event alerts again.
async def watch_model_status() -> None:
    """Poll the status file and alert only on WARNING/CRITICAL transitions."""
    last_alerted_status: Optional[str] = None
    logger.info(
        "Model status watcher started (poll interval: %ss).",
        MODEL_STATUS_POLL_SECONDS,
    )

    while True:
        try:
            current_status = await asyncio.to_thread(
                read_model_status
            )
            if current_status in {"WARNING", "CRITICAL"}:
                if current_status != last_alerted_status:
                    await asyncio.to_thread(
                        send_support_email, current_status
                    )
                    last_alerted_status = current_status
            else:
                last_alerted_status = None
        except asyncio.CancelledError:
            raise
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive guard
            logger.error(
                "Model status watcher iteration failed: %s",
                exc,
            )

        await asyncio.sleep(MODEL_STATUS_POLL_SECONDS)

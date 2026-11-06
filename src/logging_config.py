"""
===============================================================================
PharmaSupply-ML: Structured JSON Logging
===============================================================================

PURPOSE
-------
Emit machine-readable single-line JSON logs so that container stdout can be
ingested by log collectors (Loki, Elasticsearch, CloudWatch, etc.) without
custom parsing rules.

RESPONSIBILITIES
----------------
* ``JsonFormatter`` serialises level, logger, message, timestamp and selected
  request-scoped fields (request id, method, path, status, duration).
* ``configure_logging`` installs the formatter exactly once on the root logger,
  avoiding duplicated handlers when the application is imported repeatedly.

INPUTS / OUTPUTS
----------------
* Inputs: standard ``logging.LogRecord`` objects.
* Outputs: JSON strings written to stdout. Exceptions are rendered with their
  formatted traceback in the ``exception`` field.

ARCHITECTURE INTEGRATION
------------------------
Called by ``src.api.create_app`` at startup. The observability middleware logs
through this formatter; pipeline scripts keep their human-readable logging.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

# Fields injected via ``logger.info(..., extra={...})``. Only these keys are
# promoted to the top level of the JSON object to avoid leaking arbitrary data.
_EXTRA_FIELDS = (
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "event",
)


# Formatter used for both application logs and HTTP access logs.
class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(
                record.exc_info
            )
        return json.dumps(
            payload, ensure_ascii=False, default=str
        )


# Idempotent root-logger configuration. The guard prevents duplicate JSON
# handlers when tests import the API module multiple times.
def configure_logging(level: int = logging.INFO) -> None:
    """Install a JSON handler on the root logger exactly once."""
    root = logging.getLogger()
    root.setLevel(level)
    if any(
        isinstance(handler.formatter, JsonFormatter)
        for handler in root.handlers
    ):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

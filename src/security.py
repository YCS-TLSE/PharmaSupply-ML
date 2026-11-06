"""
===============================================================================
PharmaSupply-ML: API Security
===============================================================================

PURPOSE
-------
Provide the minimum production security baseline for the serving layer:
API-key authentication, explicit CORS policy and per-route rate limiting.

RESPONSIBILITIES
----------------
* ``require_api_key`` validates the ``X-API-Key`` header when ``API_KEY`` is
  configured; an empty value intentionally disables auth for local development.
* ``limiter`` applies slowapi rate limits to expensive routes (/predict, /alerts).
* ``configure_security`` installs CORS middleware and the 429 exception handler.

INPUTS / OUTPUTS
----------------
* Inputs: request headers, client IP, ``src.config`` environment settings.
* Outputs: ``401`` for a missing/invalid key, ``429`` when a quota is exceeded,
  CORS response headers for allowed dashboard origins.

ARCHITECTURE INTEGRATION
------------------------
Called once by ``src.api.create_app``. Routers declare
``dependencies=[Depends(require_api_key)]`` and decorate expensive handlers
with ``@limiter.limit(...)``.

OPERATIONAL NOTES
-----------------
The default in-memory limiter is per-process; a multi-instance deployment
should use a shared backend (Redis). The static API key is a baseline, not a
replacement for OAuth2/OIDC and secret rotation in a public deployment.

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

from __future__ import annotations

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Security,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import src.config as config

# ``auto_error=False`` lets us return a controlled 401 message instead of
# FastAPI's default 403 when the header is absent.
api_key_header = APIKeyHeader(
    name=config.API_KEY_HEADER, auto_error=False
)

# One limiter per process. Route-specific quotas are applied by decorators in
# the routers; the default limit list is intentionally empty.
limiter = Limiter(
    key_func=get_remote_address, default_limits=[]
)


# Dependency shared by every business router. Returning ``None`` when auth is
# disabled keeps the signature stable for tests and local development.
async def require_api_key(
    api_key: str | None = Security(api_key_header),
) -> str | None:
    """Validate X-API-Key when API_KEY is configured.

    When ``API_KEY`` is empty, authentication is disabled for local
    development and a warning is expected at startup.
    """
    if not config.API_KEY:
        return None
    if api_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return api_key


# Slowapi raises ``RateLimitExceeded``; translate it into a stable JSON 429
# contract instead of leaking the library's default response.
def _rate_limit_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "Rate limit exceeded. Please retry later."
        },
    )


# Application-level wiring: CORS first, then the limiter state and handler.
def configure_security(app: FastAPI) -> None:
    """Install CORS, the rate limiter and its exception handler."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.CORS_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded, _rate_limit_handler
    )

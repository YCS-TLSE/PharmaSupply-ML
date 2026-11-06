"""
HTTP routers for the PharmaSupply-ML FastAPI application.

Each module owns one API concern and is mounted by ``src.api.create_app``:

* ``health``  — model health, liveness and readiness probes.
* ``data``    — store and product catalogue lookups.
* ``alerts``  — inventory audit and prioritized operational alerts.
* ``features``— recent features and sales history for a product.
* ``predict`` — demand inference and replenishment calculation.

Routers only orchestrate HTTP concerns; business logic lives in
``src.services`` and SQL lives in ``src.repositories``.
"""

"""
===============================================================================
PharmaSupply-ML: Inventory Read Queries
===============================================================================

PURPOSE
-------
Encapsulate the read-only SQL used by the serving layer. Keeping SQL here
instead of in routers makes queries testable with stub cursors and ensures
every query benefits from the shared pooled connection.

RESPONSIBILITIES
----------------
* List distinct stores and products.
* Fetch the latest row per product for alert computation.
* Fetch the most recent feature rows for a store/product pair.

INPUTS / OUTPUTS
----------------
Inputs: store candidates and product identifiers supplied by services.
Outputs: lists of scalar IDs or dictionaries ready for the service layer.

ARCHITECTURE INTEGRATION
------------------------
Called by ``src.api_routers.data``, ``src.api_routers.alerts`` and
``src.api_routers.features``. Database failures bubble up as
``DatabaseUnavailableError`` from ``db_cursor``.
"""

from __future__ import annotations

from typing import Any

from src.repositories.database import db_cursor


# Used by /stores. Empty result is legitimate and returned as an empty list.
def fetch_store_ids() -> list[str]:
    """Return distinct store identifiers ordered alphabetically."""
    with db_cursor() as cursor:
        cursor.execute(
            "SELECT DISTINCT store_id FROM raw_pharmaceutical_demand "
            "ORDER BY store_id;"
        )
        return [row[0] for row in cursor.fetchall()]


# Used by /stores/{id}/products. The store variants are normalised upstream.
def fetch_product_ids(
    store_candidates: list[str],
) -> list[str]:
    """Return distinct product identifiers for the given store variants."""
    query = """
        SELECT DISTINCT product_id
        FROM raw_pharmaceutical_demand
        WHERE store_id::text = ANY(%s)
        ORDER BY product_id;
    """
    with db_cursor() as cursor:
        cursor.execute(query, (store_candidates,))
        return [row[0] for row in cursor.fetchall()]


# Used by /stores/{id}/alerts. DISTINCT ON keeps only the most recent row per
# product, which is the correct basis for a current inventory audit.
def fetch_alert_rows(
    store_candidates: list[str],
) -> list[dict[str, Any]]:
    """Return the latest row per product for the given stores."""
    query = """
        SELECT DISTINCT ON (product_id)
               product_id, stock_on_hand, target_demand, safety_threshold,
               critical_expiry_days, shelf_life_days
        FROM raw_pharmaceutical_demand
        WHERE store_id::text = ANY(%s)
        ORDER BY product_id, date DESC;
    """
    with db_cursor(dict_rows=True) as cursor:
        cursor.execute(query, (store_candidates,))
        return [dict(row) for row in cursor.fetchall()]


# Used by /features. Rows are returned newest-first; the service reverses them
# to reconstruct a chronological sales history.
def fetch_feature_rows(
    store_candidates: list[str],
    product_id: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Return the most recent feature rows for a store/product pair."""
    query = """
        SELECT stock_on_hand, supplier_lead_time, sales_volume,
               safety_threshold, batch_size, moq, shelf_life_days,
               critical_expiry_days
        FROM raw_pharmaceutical_demand
        WHERE store_id::text = ANY(%s) AND product_id::text = %s
        ORDER BY date DESC
        LIMIT %s;
    """
    with db_cursor(dict_rows=True) as cursor:
        cursor.execute(
            query,
            (store_candidates, str(product_id), limit),
        )
        return [dict(row) for row in cursor.fetchall()]

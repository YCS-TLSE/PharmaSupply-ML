"""
===============================================================================
PharmaSupply-ML: Synthetic Pharmaceutical Demand Generator
===============================================================================

PURPOSE
-------
Generate a realistic multi-site, multi-product daily demand history when real
pharmaceutical data cannot be shared for privacy or contractual reasons.

RESPONSIBILITIES
----------------
* Simulate trend, seasonality, weekends, stockouts and variable lead times.
* Assign per-product logistics constraints (batch size, MOQ, shelf life).
* Inject intentional anomalies and missing values to exercise validation.
* Simulate the remaining life of the most urgent batch for FEFO logic.

INPUTS / OUTPUTS
----------------
Inputs: number of days, products, stores and a random seed.
Outputs: a DataFrame (and optionally a CSV) with one row per store/product/day.

ARCHITECTURE INTEGRATION
------------------------
Consumed by ``run_pipeline.py`` before ingestion. The generated schema matches
``sql/schema.sql`` and the model feature contract. The dataset is synthetic by
design; production adapters can replace it without changing downstream code.

OPERATIONAL NOTES
-----------------
The random seed makes value generation reproducible, but the date range is
anchored to ``datetime.now()``, so runs on different days produce different
calendar indexes. This is intentional for demo freshness.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging to MLOps standard output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Main generator. The loops intentionally inject MED_003/MED_008 anomalies and
# NaN logistics values so the validation gate and alert engine are exercised.
def build_pharma_dataset(
    days: int = 1095,
    num_products: int = 10,
    store_ids: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate 3 years of historical daily demand, stock levels, safety thresholds,
    and logistics constraints across multiple pharmacy stores.
    """
    if store_ids is None:
        store_ids = ["1", "2", "3"]

    np.random.seed(seed)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    dates = [
        start_date + timedelta(days=d) for d in range(days)
    ]
    products = [
        f"MED_{i + 1:03d}" for i in range(num_products)
    ]

    rows = []

    # Iterate over each store to generate independent product inventories and sales histories
    for store_id in store_ids:
        for product in products:
            base_demand = np.random.randint(50, 150)
            current_stock = np.random.randint(800, 2000)

            # --- DYNAMIC SAFETY THRESHOLD ASSIGNMENT & ANOMALY INJECTION ---
            default_safety = int(base_demand * 4)

            # Inject store-specific operational anomalies to test multi-site alert filtering
            if product == "MED_003" and store_id == "1":
                # Test case 1: Intentionally TOO LOW safety threshold for a high-volume product
                safety_threshold = 150.0
                batch_size = 100
                moq = 500
                shelf_life_days = 365
            elif product == "MED_008":
                # Test case 2: Intentionally TOO HIGH safety threshold (stock over-immobilization)
                safety_threshold = 4500.0
                batch_size = 50
                moq = 200
                shelf_life_days = (
                    180  # Shorter shelf life test case
                )
            else:
                safety_threshold = float(default_safety)

                # --- CUSTOM BATCH SIZE & MOQ PER PRODUCT WITH INTENTIONAL MISSING VALUES FOR TESTING ---
                if product in ["MED_001", "MED_002"]:
                    batch_size = 50
                    moq = 100
                elif product in ["MED_004", "MED_005"]:
                    batch_size = 250
                    moq = 500
                else:
                    # Intentionally missing values (NaN) to test validation pipeline resilience
                    batch_size = None
                    moq = None

                shelf_life_days = 365

            for day_idx, current_date in enumerate(dates):
                # 1. Trend & Seasonality
                annual_trend = 1.0 + (
                    0.04 * (day_idx / 365)
                )

                is_winter = current_date.month in [
                    11,
                    12,
                    1,
                    2,
                ]
                is_spring = current_date.month in [3, 4, 5]

                seasonality = (
                    1.30
                    if is_winter
                    else (1.15 if is_spring else 1.0)
                )

                is_weekend = current_date.weekday() >= 5
                weekend_scale = 0.50 if is_weekend else 1.0

                raw_demand = (
                    np.random.normal(base_demand, 10)
                    * seasonality
                    * weekend_scale
                    * annual_trend
                )
                demand = max(0, int(raw_demand))

                # 2. Stockout check & Censored sales
                is_stockout = int(current_stock < demand)
                fulfilled_sales = min(current_stock, demand)
                current_stock -= fulfilled_sales

                # 3. Variable supplier lead time & Supply shocks
                lead_time = int(
                    np.random.choice(
                        [2, 3, 5, 8, 12],
                        p=[0.50, 0.30, 0.10, 0.07, 0.03],
                    )
                )

                if current_stock < 300:
                    current_stock += np.random.randint(
                        500, 1200
                    )

                # --- 4. CRITICAL EXPIRY TRACKING (FEFO PREPARATION) ---
                # Simulates the remaining days before expiration of the most urgent active batch.
                if product == "MED_008":
                    critical_expiry_days = max(
                        5,
                        int(
                            shelf_life_days
                            - (
                                (day_idx * 3)
                                % shelf_life_days
                            )
                        ),
                    )
                else:
                    critical_expiry_days = max(
                        30,
                        int(
                            shelf_life_days
                            - (
                                (day_idx * 1.5)
                                % shelf_life_days
                            )
                        ),
                    )

                rows.append(
                    {
                        "date": current_date.strftime(
                            "%Y-%m-%d"
                        ),
                        "store_id": str(store_id),
                        "product_id": product,
                        "target_demand": demand,
                        "sales_volume": fulfilled_sales,
                        "stock_on_hand": current_stock,
                        "is_stockout": is_stockout,
                        "supplier_lead_time": lead_time,
                        "safety_threshold": safety_threshold,
                        "batch_size": batch_size,
                        "moq": moq,
                        "shelf_life_days": shelf_life_days,
                        "critical_expiry_days": critical_expiry_days,
                    }
                )

    return pd.DataFrame(rows)


# CLI entry point: generate the demo dataset and persist it as CSV.
if __name__ == "__main__":
    logging.info(
        "Generating multi-site 3-year synthetic pharmaceutical dataset..."
    )
    df_pharma = build_pharma_dataset(
        store_ids=["1", "2", "3"]
    )

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir / "pharmaceutical_demand_row.csv"
    )
    df_pharma.to_csv(output_file, index=False)

    logging.info(
        f"Multi-site dataset generated successfully ({len(df_pharma)} rows) -> {output_file}"
    )

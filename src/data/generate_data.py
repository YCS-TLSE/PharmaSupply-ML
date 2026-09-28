"""Module for generating synthetic dataset for pharmaceutical demand forecasting.

CONTEXT & PURPOSE:
------------------
Due to data privacy restrictions and the lack of immediate access to real-world
pharmaceutical sales databases during the project setup phase, this script generates
a realistic 3-year synthetic dataset (1,095 days).

It mimics supply chain dynamics:
1. Multi-year seasonality (winter flu spikes, spring allergies) and market growth trend.
2. Censored demand during active stockouts (sales_volume != target_demand).
3. Variable supplier lead times with probabilistic delivery shocks.

DATA STRUCTURE:
---------------
- Row Count: 10,950 rows (1,095 days x 10 pharmaceutical products)
- Features Generated:
    1. date (str): YYYY-MM-DD timestamp.
    2. product_id (str): Unique product code (e.g., MED_001).
    3. target_demand (int): True daily customer demand.
    4. sales_volume (int): Actual sales fulfilled based on available stock.
    5. stock_on_hand (int): Daily remaining inventory.
    6. is_stockout (int): Binary flag (1 if stock < target demand, else 0).
    7. supplier_lead_time (int): Simulated delivery delay in days.
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


def build_pharma_dataset(days: int = 1095, num_products: int = 10, seed: int = 42) -> pd.DataFrame:
    """Generate 3 years of historical daily demand and stock levels."""
    np.random.seed(seed)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    dates = [start_date + timedelta(days=d) for d in range(days)]
    products = [f"MED_{i + 1:03d}" for i in range(num_products)]

    rows = []

    for product in products:
        base_demand = np.random.randint(50, 150)
        current_stock = np.random.randint(800, 2000)

        for day_idx, current_date in enumerate(dates):
            # 1. Trend & Seasonality
            annual_trend = 1.0 + (0.04 * (day_idx / 365))  # +4% growth per year

            is_winter = current_date.month in [11, 12, 1, 2]
            is_spring = current_date.month in [3, 4, 5]

            seasonality = 1.30 if is_winter else (1.15 if is_spring else 1.0)

            # Weekend effect (lower order volumes)
            is_weekend = current_date.weekday() >= 5
            weekend_scale = 0.50 if is_weekend else 1.0

            # Compute true target demand
            raw_demand = (
                np.random.normal(base_demand, 10) * seasonality * weekend_scale * annual_trend
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

            # Automatic replenishment trigger
            if current_stock < 300:
                current_stock += np.random.randint(500, 1200)

            rows.append(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "product_id": product,
                    "target_demand": demand,
                    "sales_volume": fulfilled_sales,
                    "stock_on_hand": current_stock,
                    "is_stockout": is_stockout,
                    "supplier_lead_time": lead_time,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.info("Generating 3-year synthetic pharmaceutical dataset...")
    df_pharma = build_pharma_dataset()

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "pharmaceutical_demand_row.csv"
    df_pharma.to_csv(output_file, index=False)

    logging.info(f"Dataset generated successfully ({len(df_pharma)} rows) -> {output_file}")

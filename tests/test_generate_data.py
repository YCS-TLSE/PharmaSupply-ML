"""
Synthetic dataset generator tests.

Checks the generated schema, row count, non-negative demand/sales and the
multi-store/product contract expected by ingestion and feature engineering.
"""

from __future__ import annotations

from src.data.generate_data import build_pharma_dataset


def test_build_pharma_dataset_shape_and_contract():
    frame = build_pharma_dataset(
        days=40,
        num_products=2,
        store_ids=["1"],
        seed=42,
    )
    assert len(frame) == 80
    expected_columns = {
        "date",
        "store_id",
        "product_id",
        "target_demand",
        "sales_volume",
        "stock_on_hand",
        "is_stockout",
        "supplier_lead_time",
        "safety_threshold",
        "batch_size",
        "moq",
        "shelf_life_days",
        "critical_expiry_days",
    }
    assert expected_columns.issubset(frame.columns)
    assert (frame["sales_volume"] >= 0).all()
    assert (frame["target_demand"] >= 0).all()
    assert frame["store_id"].unique().tolist() == ["1"]
    assert frame["product_id"].nunique() == 2

"""Initial raw pharmaceutical demand schema.

This migration creates the staging table used by ingestion, feature
engineering and the API. The natural key ``(store_id, product_id, date)`` is
declared as a unique constraint so the ingestion UPSERT can target it.

Revision ID: 0001
Revises:
Create Date: 2026-11-06
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


# Create the staging table with explicit types, constrained logistics fields and
# the natural business key. Defaults mirror the validation/ingestion defaults.
def upgrade() -> None:
    op.create_table(
        "raw_pharmaceutical_demand",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column(
            "store_id", sa.String(length=50), nullable=False
        ),
        sa.Column(
            "product_id",
            sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            "target_demand", sa.Integer(), nullable=False
        ),
        sa.Column(
            "sales_volume", sa.Integer(), nullable=False
        ),
        sa.Column(
            "stock_on_hand", sa.Integer(), nullable=False
        ),
        sa.Column(
            "is_stockout",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "supplier_lead_time",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "safety_threshold",
            sa.Numeric(10, 2),
            nullable=False,
            server_default="150.0",
        ),
        sa.Column(
            "batch_size",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
        sa.Column(
            "moq",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
        sa.Column(
            "shelf_life_days",
            sa.Integer(),
            nullable=False,
            server_default="365",
        ),
        sa.Column(
            "critical_expiry_days",
            sa.Integer(),
            nullable=False,
            server_default="365",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint(
            "store_id",
            "product_id",
            "date",
            name="uq_pharma_store_product_date",
        ),
    )
    op.create_index(
        "idx_pharma_store_product_date",
        "raw_pharmaceutical_demand",
        ["store_id", "product_id", "date"],
    )


# Revert the schema in reverse dependency order (index first, then table).
def downgrade() -> None:
    op.drop_index(
        "idx_pharma_store_product_date",
        table_name="raw_pharmaceutical_demand",
    )
    op.drop_table("raw_pharmaceutical_demand")

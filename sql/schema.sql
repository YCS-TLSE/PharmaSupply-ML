-- Schema DDL for PharmaSupply-ML (Staging / Raw Data)
-- Designed for a central purchasing and distribution hub with real-world logistical and shelf-life constraints.

DROP TABLE IF EXISTS raw_pharmaceutical_demand CASCADE;

CREATE TABLE raw_pharmaceutical_demand (
    -- Unique identifier for each tracking record/row
    id SERIAL PRIMARY KEY,
    
    -- Date of the record (Time-series dimension)
    date DATE NOT NULL,
    
    -- Identifier for the target warehouse, hub, or regional store
    store_id VARCHAR(50) NOT NULL,
    
    -- Identifier for the pharmaceutical product/reference
    product_id VARCHAR(50) NOT NULL,
    
    -- Machine Learning target: predicted or actual future demand to cover
    target_demand INT NOT NULL,
    
    -- Actual historical sales volume recorded for the period
    sales_volume INT NOT NULL,
    
    -- Current physical stock level available on-hand in the warehouse
    stock_on_hand INT NOT NULL,
    
    -- Binary flag indicating whether a stockout event occurred (1 = Yes, 0 = No)
    is_stockout INT NOT NULL DEFAULT 0,
    
    -- Supplier lead time expressed in days (time required for delivery from supplier)
    supplier_lead_time INT NOT NULL,
    
    -- Minimum safety stock threshold triggering replenishment alerts
    safety_threshold NUMERIC(10, 2) NOT NULL DEFAULT 150.0,
    -- Logistics constraint: packaging multiple / box size (e.g., must order in multiples of 50 or 100 units)
    batch_size INT NOT NULL DEFAULT 1,
    
    -- Logistics constraint: Minimum Order Quantity (MOQ) enforced by the supplier to validate an order
    moq INT NOT NULL DEFAULT 1,
    
    -- Expiry constraint: Total shelf-life duration in days from manufacture, used to prevent obsolescence and waste
    shelf_life_days INT NOT NULL DEFAULT 365,
    
    -- Expiry constraint: Remaining days before expiration of the most urgent active batch (FEFO preparation)
    critical_expiry_days INT NOT NULL DEFAULT 365,
    
    -- Timestamp recording when the record was ingested into the database
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimized multi-column index for fast time-series queries across stores and products
CREATE INDEX idx_pharma_store_product_date ON raw_pharmaceutical_demand (store_id, product_id, date);
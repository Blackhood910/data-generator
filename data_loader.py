#!/usr/bin/env python3
"""
Load Alison Kingsgate synthetic CSVs into MySQL from VS Code.

What’s special in this version?
- If a CSV has columns not present in the table (e.g. products.category_name),
  we will CREATE those columns in MySQL as TEXT NULL and insert NULLs (not the CSV values).
  This keeps schema compatibility without storing that data.
- NaN/NaT/Inf and stringy "nan"/"None"/"" -> SQL NULL
- Booleans -> tinyint(1)
- INSERT IGNORE for idempotent re-runs
- Column order auto-aligned to table schema

Usage:
  python data_loader.py ^
    --base-dir .\ag_data\clean ^
    --host 127.0.0.1 --user root --password "YOURPASS" ^
    --database MSk_e_com_AKGate
"""

import os
import argparse
import pandas as pd
import numpy as np
import mysql.connector

# ---- MySQL schema (creates DB if missing, then USE it) ----
MYSQL_DDL_TEMPLATE = """
CREATE DATABASE IF NOT EXISTS `{db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `{db}`;

CREATE TABLE IF NOT EXISTS brands (
  brand_id INT PRIMARY KEY,
  brand_name TEXT NOT NULL,
  website_url TEXT
);

CREATE TABLE IF NOT EXISTS platforms (
  platform_id INT PRIMARY KEY,
  platform_code VARCHAR(32) NOT NULL,
  platform_name VARCHAR(128) NOT NULL,
  region VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS categories (
  category_id INT PRIMARY KEY,
  parent_category_id INT NULL,
  category_name VARCHAR(128) NOT NULL,
  category_path VARCHAR(256)
);

CREATE TABLE IF NOT EXISTS products (
  product_id INT PRIMARY KEY,
  sku VARCHAR(64),
  product_name VARCHAR(256) NOT NULL,
  category_id INT,
  brand_id INT,
  actual_price DECIMAL(10,2),
  discounted_price DECIMAL(10,2),
  discount_percentage DECIMAL(6,4),
  rating DECIMAL(3,2),
  rating_count INT,
  about_product TEXT,
  img_link TEXT,
  product_link TEXT,
  status VARCHAR(32),
  unit_cost DECIMAL(10,2),
  default_list_price DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS product_variants (
  variant_id INT PRIMARY KEY,
  product_id INT,
  variant_sku VARCHAR(64),
  size_code VARCHAR(32),
  width_mm INT, height_mm INT,
  frame_material VARCHAR(64),
  frame_finish VARCHAR(64),
  frame_profile VARCHAR(64),
  glazing_type VARCHAR(64),
  mount_included_flag TINYINT(1),
  mount_color VARCHAR(64),
  backing_type VARCHAR(64),
  orientation VARCHAR(32),
  unit_cost DECIMAL(10,2),
  default_list_price DECIMAL(10,2),
  weight_kg DECIMAL(8,2),
  package_length_mm INT,
  package_width_mm INT,
  package_height_mm INT,
  status VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS marketplace_accounts (
  account_id INT PRIMARY KEY,
  platform_id INT,
  merchant_slug VARCHAR(128),
  default_currency VARCHAR(8)
);

CREATE TABLE IF NOT EXISTS product_listings (
  listing_id INT PRIMARY KEY,
  product_id INT,
  variant_id INT NULL,
  platform_id INT,
  account_id INT,
  listing_sku VARCHAR(128),
  title VARCHAR(256),
  subtitle VARCHAR(256),
  description_html TEXT,
  bullets_json JSON,
  main_image_url TEXT,
  additional_images_json JSON,
  amazon_asin VARCHAR(16),
  amazon_marketplace_id VARCHAR(32),
  amazon_fulfilment_channel VARCHAR(8),
  ebay_item_id VARCHAR(32),
  ebay_listing_type VARCHAR(32),
  ebay_condition_id INT,
  ebay_category_id INT,
  is_active TINYINT(1)
);

CREATE TABLE IF NOT EXISTS listing_prices (
  price_id INT PRIMARY KEY,
  listing_id INT,
  currency VARCHAR(8),
  listing_price DECIMAL(10,2),
  sale_price DECIMAL(10,2),
  valid_from DATE,
  valid_to DATE NULL
);

CREATE TABLE IF NOT EXISTS platform_fees (
  platform_fee_id INT PRIMARY KEY,
  platform_id INT,
  fee_type VARCHAR(64),
  fee_percent DECIMAL(6,4),
  fee_flat_amount DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS channel_inventory (
  channel_inventory_id INT PRIMARY KEY,
  listing_id INT,
  on_hand_qty INT,
  reserved_qty INT,
  backorder_qty INT
);

CREATE TABLE IF NOT EXISTS customers (
  customer_id INT PRIMARY KEY,
  first_name VARCHAR(64),
  last_name VARCHAR(64),
  email VARCHAR(256),
  phone VARCHAR(64),
  gender VARCHAR(32),
  age_group VARCHAR(32),
  region VARCHAR(64),
  signup_source VARCHAR(64),
  preferred_platform VARCHAR(16),
  repeat_customer_flag TINYINT(1)
);

CREATE TABLE IF NOT EXISTS orders (
  order_id INT PRIMARY KEY,
  order_number VARCHAR(64),
  order_date DATETIME,
  platform_id INT,
  account_id INT,
  customer_id INT,
  currency VARCHAR(8),
  subtotal_amount DECIMAL(12,2),
  discount_amount DECIMAL(12,2),
  tax_amount DECIMAL(12,2),
  shipping_amount DECIMAL(12,2),
  channel_fee_amount DECIMAL(12,2),
  total_amount DECIMAL(12,2),
  order_status VARCHAR(32),
  delivery_days INT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
  order_item_id INT PRIMARY KEY,
  order_id INT,
  line_number INT,
  product_id INT,
  variant_id INT,
  listing_id INT,
  quantity INT,
  unit_price DECIMAL(12,2),
  line_subtotal DECIMAL(12,2),
  line_discount DECIMAL(12,2),
  line_tax DECIMAL(12,2),
  line_total DECIMAL(12,2),
  unit_cost DECIMAL(12,2),
  margin_amount DECIMAL(12,2)
);

CREATE TABLE IF NOT EXISTS order_fees (
  order_fee_id INT PRIMARY KEY,
  order_id INT,
  platform_id INT,
  fee_type VARCHAR(64),
  fee_amount DECIMAL(12,2)
);

CREATE TABLE IF NOT EXISTS payments (
  payment_id INT PRIMARY KEY,
  order_id INT,
  payment_method VARCHAR(32),
  provider_txn_id VARCHAR(64),
  amount DECIMAL(12,2),
  status VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS shipments (
  shipment_id INT PRIMARY KEY,
  order_id INT,
  carrier VARCHAR(64),
  tracking_number VARCHAR(64),
  shipped_at DATETIME,
  delivered_at DATETIME NULL,
  delivery_status VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS returns (
  return_id INT PRIMARY KEY,
  order_id INT,
  return_number VARCHAR(64),
  status VARCHAR(32),
  initiated_at DATETIME
);

CREATE TABLE IF NOT EXISTS return_items (
  return_item_id INT PRIMARY KEY,
  return_id INT,
  order_item_id INT,
  quantity_returned INT,
  return_reason VARCHAR(64),
  refund_amount DECIMAL(12,2)
);

CREATE TABLE IF NOT EXISTS reviews (
  review_id VARCHAR(32) PRIMARY KEY,
  product_id INT,
  variant_id INT NULL,
  source_platform VARCHAR(16),
  user_id VARCHAR(32),
  user_name VARCHAR(128),
  review_title VARCHAR(256),
  review_content TEXT,
  rating INT
);

CREATE TABLE IF NOT EXISTS warehouses (
  warehouse_id INT PRIMARY KEY,
  warehouse_code VARCHAR(32),
  warehouse_name VARCHAR(128),
  city VARCHAR(128),
  country VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS inventory (
  inventory_id INT PRIMARY KEY,
  variant_id INT,
  warehouse_id INT,
  on_hand_qty INT,
  reserved_qty INT,
  reorder_point INT,
  safety_stock INT
);
"""

LOAD_ORDER = [
    "brands","platforms","categories","platform_fees","marketplace_accounts",
    "products","product_variants","product_listings","listing_prices","channel_inventory",
    "customers",
    "orders","order_items","order_fees","payments","shipments","returns","return_items",
    "reviews",
    "warehouses","inventory",
]

BOOL_COLS = {
    "product_variants": ["mount_included_flag"],
    "product_listings": ["is_active"],
    "customers": ["repeat_customer_flag"],
}

def normalise_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all NaN/NaT/Inf and stringy nulls to Python None so MySQL gets real NULLs."""
    df = df.astype(object)
    df = df.replace({
        np.nan: None, np.inf: None, -np.inf: None,
        "nan": None, "NaN": None, "None": None, "": None
    })
    return df

def to_bool_int(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true","1","t","yes","y"]).astype(int)
    return df

def get_table_columns(cursor, table: str):
    """Return ordered list of column names for an existing MySQL table."""
    cursor.execute(f"SHOW COLUMNS FROM `{table}`;")
    return [row[0] for row in cursor.fetchall()]

def ensure_extra_columns_as_null(cursor, table: str, extra_cols):
    """
    For any CSV columns not present in the table, add them as TEXT NULL,
    and we will insert NULLs for them (not the CSV values).
    """
    if not extra_cols:
        return
    for col in extra_cols:
        safe = col.replace("`", "``")
        cursor.execute(f"ALTER TABLE `{table}` ADD COLUMN `{safe}` TEXT NULL;")
    print(f"[INFO] {table}: created extra columns as NULL {list(extra_cols)}")

def align_df_to_table(cursor, table: str, df: pd.DataFrame, add_extras_as_null=True) -> pd.DataFrame:
    """
    - If add_extras_as_null: add any extra CSV columns to table (TEXT NULL) and set df[col]=None
      (values are discarded), then align/reorder.
    - Else: drop extras.
    - Always add any missing table columns to df as None.
    """
    table_cols = get_table_columns(cursor, table)
    extra = [c for c in df.columns if c not in table_cols]
    missing = [c for c in table_cols if c not in df.columns]

    if extra and add_extras_as_null:
        ensure_extra_columns_as_null(cursor, table, extra)
        # refresh table cols after ALTER TABLE
        table_cols = get_table_columns(cursor, table)
        # ensure we don't insert actual values for those extras
        for c in extra:
            df[c] = None
        # recompute missing in case new columns appeared
        missing = [c for c in table_cols if c not in df.columns]
    elif extra:
        print(f"[INFO] {table}: dropping extra columns {extra}")
        df = df[[c for c in df.columns if c in table_cols]]

    for c in missing:
        df[c] = None

    # reorder to the table's column order
    df = df[table_cols]
    return df, table_cols

def chunked(df: pd.DataFrame, size: int = 5000):
    for i in range(0, len(df), size):
        yield df.iloc[i:i+size]

def load_table(cursor, table: str, df: pd.DataFrame, add_extras_as_null=True):
    # normalise + booleans
    df = normalise_nulls(df)
    df = to_bool_int(df, BOOL_COLS.get(table, []))
    # align columns (and possibly add extras to table as NULL)
    df, cols = align_df_to_table(cursor, table, df, add_extras_as_null=add_extras_as_null)

    placeholders = ",".join(["%s"] * len(cols))
    sql = f"INSERT IGNORE INTO `{table}` ({', '.join(f'`{c}`' for c in cols)}) VALUES ({placeholders})"
    for part in chunked(df, 5000):
        cursor.executemany(sql, [tuple(x) for x in part.to_numpy()])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="./ag_data/clean", help="Folder with the CSVs (e.g., ./ag_data/clean)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3306)
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--database", required=True, help="Database name to create/use")
    ap.add_argument("--add-extras-as-null", action="store_true", default=True,
                    help="Auto-add extra CSV columns to table as TEXT NULL and insert NULLs (default on).")
    args = ap.parse_args()

    db_name = args.database.strip()

    conn = mysql.connector.connect(host=args.host, port=args.port, user=args.user, password=args.password, autocommit=False)
    cur = conn.cursor()

    # Create DB + tables and USE it
    ddl = MYSQL_DDL_TEMPLATE.format(db=db_name)
    for stmt in ddl.split(";\n"):
        s = stmt.strip()
        if s:
            cur.execute(s + ";")
    conn.commit()
    cur.execute(f"USE `{db_name}`;")

    # Load in dependency-safe order
    for name in LOAD_ORDER:
        path = os.path.join(args.base_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"[SKIP] {name} → {path} not found")
            continue
        print(f"[LOAD] {name} from {path}")
        df = pd.read_csv(path)
        load_table(cur, name, df, add_extras_as_null=args.add_extras_as_null)
        conn.commit()
        print(f"[OK] {name} ({len(df)} rows)")

    cur.close()
    conn.close()
    print("✅ Done.")

if __name__ == "__main__":
    main()

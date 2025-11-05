#!/usr/bin/env python3
"""
Generate a realistic multi-channel retail dataset for Alison Kingsgate / MSK E-COM LTD.

Changes in this version:
- Guaranteed non-null order timestamps.
- Adds split date/time columns in CLEAN export:
    * order_date_only: YYYY-MM-DD (DATE)
    * order_time_only: HH:MM:SS (TIME)
- CLEAN orders are written directly from canonical timestamps (no re-parsing).

Outputs
- <base>/raw/*.csv     (intentionally messy sources)
- <base>/clean/*.csv   (clean canonical tables)
- <base>/create_tables.sql (MySQL/PostgreSQL-friendly DDL)
- optional ZIP of everything

Example:
  python generate_ag_dataset.py --base-dir ./ag_data --orders 50000 --customers 20000 --products 300 --reviews 50000 --seed 123 --zip
"""
import os, json, random, string, zipfile, argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ---------- helpers ----------
def rand_date(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def rchoice_w(options, weights):
    return random.choices(options, weights=weights, k=1)[0]

def asin():  return "B0" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
def ebay_id(): return "".join(random.choices(string.digits, k=12))
def slugify(t): return "".join(ch.lower() if ch.isalnum() else "-" for ch in t).strip("-")
def money(x):  return round(float(x) + 1e-9, 2)

def ensure_dirs(base):
    raw = os.path.join(base, "raw"); clean = os.path.join(base, "clean")
    os.makedirs(raw, exist_ok=True); os.makedirs(clean, exist_ok=True)
    return raw, clean

# cleaning helpers (for RAW → CLEAN where used)
def clean_money_series(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.replace("£","", regex=False)
             .str.replace(",","", regex=False).str.strip()
             .replace("", np.nan).astype(float).round(2))

def standardize_title(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.title()

def parse_mixed_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

# ---------- generator ----------
def generate_dataset(base_dir="./ag_data",
                     n_orders=50000, n_customers=20000, n_products=300, n_reviews=50000,
                     seed=123, zip_output=False):
    random.seed(seed); np.random.seed(seed)
    raw_dir, clean_dir = ensure_dirs(base_dir)

    # reference tables
    brands = pd.DataFrame([{"brand_id":1,"brand_name":"Alison Kingsgate","website_url":"https://www.alisonkingsgate.co.uk"}])
    platforms = pd.DataFrame([
        {"platform_id":1,"platform_code":"AMAZON","platform_name":"Amazon UK","region":"UK"},
        {"platform_id":2,"platform_code":"EBAY","platform_name":"eBay UK","region":"UK"},
        {"platform_id":3,"platform_code":"WEBSITE","platform_name":"AlisonKingsgate.co.uk","region":"UK"},
    ])
    categories = pd.DataFrame([
        {"category_id":1,"parent_category_id":None,"category_name":"Picture Frames","category_path":"Home > Frames"},
        {"category_id":2,"parent_category_id":1,"category_name":"Poster Frames","category_path":"Home > Frames > Poster"},
        {"category_id":3,"parent_category_id":1,"category_name":"Photo Frames","category_path":"Home > Frames > Photo"},
        {"category_id":4,"parent_category_id":1,"category_name":"Certificate Frames","category_path":"Home > Frames > Certificate"},
        {"category_id":5,"parent_category_id":1,"category_name":"Collage Frames","category_path":"Home > Frames > Collage"},
    ])
    marketplace_accounts = pd.DataFrame([
        {"account_id":1,"platform_id":1,"merchant_slug":"alisonkingsgate","default_currency":"GBP"},
        {"account_id":2,"platform_id":2,"merchant_slug":"alisonkingsgate","default_currency":"GBP"},
        {"account_id":3,"platform_id":3,"merchant_slug":"alisonkingsgate","default_currency":"GBP"},
    ])
    platform_fees = pd.DataFrame([
        {"platform_fee_id":1,"platform_id":1,"fee_type":"Referral","fee_percent":0.15,"fee_flat_amount":0.0},
        {"platform_fee_id":2,"platform_id":2,"fee_type":"FinalValueFee","fee_percent":0.12,"fee_flat_amount":0.0},
        {"platform_fee_id":3,"platform_id":3,"fee_type":"Gateway","fee_percent":0.015,"fee_flat_amount":0.2},
    ])

    # catalog
    frame_materials = ["MDF","Solid Wood","Aluminium"]
    frame_finishes  = ["Black","White","Oak","Walnut","Gold","Silver","Rustic"]
    frame_profiles  = ["Modern","Classic","Slim","Box"]
    glazing_types   = ["Acrylic","Perspex","Glass"]
    mount_colors    = ["White","Ivory","Black","No Mount"]
    orientations    = ["Portrait","Landscape"]
    size_codes      = ["A1","A2","A3","A4","5x7","8x10","12x16","16x20","24x36"]
    size_dims_mm    = {"A1":(594,841),"A2":(420,594),"A3":(297,420),"A4":(210,297),
                       "5x7":(127,178),"8x10":(203,254),"12x16":(305,406),"16x20":(406,508),"24x36":(610,914)}

    def prod_name():
        return f"{random.choice(frame_finishes)} {random.choice(frame_profiles)} {random.choice(['Picture Frame','Photo Frame','Poster Frame','Certificate Frame'])}"

    products = []
    for pid in range(1, n_products+1):
        name = prod_name()
        cat_id = int(categories.sample(1).category_id.iloc[0])
        unit_cost  = money(np.random.uniform(4,45))
        list_price = money(unit_cost*np.random.uniform(1.6,2.8))
        disc_pct   = max(0, min(0.4, np.random.normal(0.12,0.08)))
        discounted = money(list_price*(1-disc_pct))
        rating     = float(np.clip(np.random.normal(4.3,0.4), 1.0, 5.0))
        rating_ct  = max(0, int(np.random.poisson(80)))
        products.append({
            "product_id": pid, "sku": f"AK-{pid:04d}", "product_name": name,
            "category_id": cat_id, "brand_id": 1,
            "actual_price": list_price, "discounted_price": discounted,
            "discount_percentage": round(1 - discounted/list_price, 4) if list_price else 0.0,
            "rating": round(rating,2), "rating_count": rating_ct,
            "about_product": f"High-quality {name.lower()} suitable for home and office décor. Includes hanging hardware.",
            "img_link": f"https://cdn.example/{slugify(name)}-{pid}.jpg",
            "product_link": f"https://www.alisonkingsgate.co.uk/products/{slugify(name)}-{pid}",
            "status":"ACTIVE","unit_cost":unit_cost,"default_list_price":list_price
        })
    products_df = pd.DataFrame(products)

    variants = []
    vid = 1
    for _, p in products_df.iterrows():
        for sc in random.sample(size_codes, random.choice([2,3,3,4])):
            w,h = size_dims_mm[sc]
            variants.append({
                "variant_id": vid, "product_id": int(p.product_id),
                "variant_sku": f"{p.sku}-{sc}", "size_code": sc,
                "width_mm": w, "height_mm": h,
                "frame_material": rchoice_w(frame_materials,[0.6,0.3,0.1]),
                "frame_finish": random.choice(frame_finishes),
                "frame_profile": random.choice(frame_profiles),
                "glazing_type": rchoice_w(glazing_types,[0.6,0.25,0.15]),
                "mount_included_flag": random.random()<0.6,
                "mount_color": random.choice(mount_colors),
                "backing_type": random.choice(["MDF Board","Foam Board","Card Backing"]),
                "orientation": random.choice(orientations),
                "unit_cost": money(p.unit_cost*(0.8+random.random()*0.6)),
                "default_list_price": money(p.default_list_price*(0.8+random.random()*0.6)),
                "weight_kg": round(np.random.uniform(0.4,3.5),2),
                "package_length_mm": w+30, "package_width_mm": h+30, "package_height_mm": random.choice([30,40,50]),
                "status":"ACTIVE"
            })
            vid += 1
    variants_df = pd.DataFrame(variants)

    # listings, prices, channel inventory
    listings, listing_prices, channel_inventory = [], [], []
    lid = 1
    for _, p in products_df.iterrows():
        for _, plat in platforms.iterrows():
            base_price = p.discounted_price * (1.02 if plat.platform_code=="AMAZON" else 0.99 if plat.platform_code=="EBAY" else 1.00)
            listings.append({
                "listing_id": lid, "product_id": int(p.product_id), "variant_id": None,
                "platform_id": int(plat.platform_id), "account_id": int(plat.platform_id),
                "listing_sku": f"{p.sku}-{plat.platform_code}", "title": p.product_name, "subtitle":"",
                "description_html": f"<p>{p.about_product}</p>",
                "bullets_json": json.dumps(["Ready to hang","Multiple sizes","UK dispatch"]),
                "main_image_url": p.img_link, "additional_images_json": json.dumps([p.img_link]),
                "amazon_asin": asin() if plat.platform_code=="AMAZON" else None,
                "amazon_marketplace_id":"A1F83G8C2ARO7P" if plat.platform_code=="AMAZON" else None,
                "amazon_fulfilment_channel": random.choice(["FBA","FBM"]) if plat.platform_code=="AMAZON" else None,
                "ebay_item_id": ebay_id() if plat.platform_code=="EBAY" else None,
                "ebay_listing_type": "FixedPrice" if plat.platform_code=="EBAY" else None,
                "ebay_condition_id": 1000 if plat.platform_code=="EBAY" else None,
                "ebay_category_id": 156389 if plat.platform_code=="EBAY" else None,
                "is_active": True
            })
            listing_prices += [
                {"price_id": len(listing_prices)+1,"listing_id": lid,"currency":"GBP",
                 "listing_price": money(base_price),"sale_price": money(base_price),
                 "valid_from":"2025-06-01","valid_to":"2025-09-01"},
                {"price_id": len(listing_prices)+1,"listing_id": lid,"currency":"GBP",
                 "listing_price": money(base_price+random.choice([0,1,2])),
                 "sale_price": money(base_price+random.choice([0,1])),
                 "valid_from":"2025-09-01","valid_to": None},
            ]
            channel_inventory.append({
                "channel_inventory_id": len(channel_inventory)+1,"listing_id": lid,
                "on_hand_qty": random.randint(5,200),"reserved_qty": random.randint(0,10),"backorder_qty": random.choice([0,0,0,1,2])
            })
            lid += 1
    listings_df = pd.DataFrame(listings)
    listing_prices_df = pd.DataFrame(listing_prices)
    channel_inventory_df = pd.DataFrame(channel_inventory)

    # customers
    first_names = ["Alex","Sam","Chris","Jordan","Taylor","Morgan","Casey","Jamie","Robin","Avery","Lee","Dana","Cameron","Riley","Jesse","Sky","Harper","Quinn","Rowan","Sage"]
    last_names  = ["Smith","Brown","Taylor","Wilson","Thomson","Anderson","Jackson","White","Harris","Martin","Clarke","Walker","Young","Wright","King","Green","Hall","Wood","Lewis","Scott"]
    regions     = ["London","Scotland","Midlands","North West","South East","Wales","Northern Ireland","South West","North East","Yorkshire"]
    age_groups  = ["18-24","25-34","35-44","45-54","55-64","65+"]
    genders     = ["F","M","Other","Prefer not to say"]

    customers = []
    for cid in range(1, n_customers+1):
        fn, ln = random.choice(first_names), random.choice(last_names)
        customers.append({
            "customer_id": cid, "first_name": fn, "last_name": ln,
            "email": f"{fn}.{ln}{cid}@example.com".lower(),
            "phone": f"+44 7{random.randint(100000000,999999999)}",
            "gender": random.choice(genders),
            "age_group": rchoice_w(age_groups,[0.1,0.25,0.22,0.2,0.15,0.08]),
            "region": random.choice(regions),
            "signup_source": random.choice(["Amazon","eBay","Website","Facebook Ads","Google Ads"]),
            "preferred_platform": rchoice_w(["AMAZON","EBAY","WEBSITE"],[0.55,0.15,0.30]),
            "repeat_customer_flag": random.random()<0.30
        })
    customers_df = pd.DataFrame(customers)

    # fast lookups
    product_ids = products_df["product_id"].values
    product_unit_cost  = products_df.set_index("product_id")["unit_cost"].to_dict()
    product_disc_price = products_df.set_index("product_id")["discounted_price"].to_dict()
    prod_variants_map  = variants_df.groupby("product_id")["variant_id"].apply(list).to_dict()
    listings_map       = listings_df.groupby(["product_id","platform_id"])["listing_id"].apply(list).to_dict()

    platform_fee_percent = {"AMAZON":0.15,"EBAY":0.12,"WEBSITE":0.015}
    platform_flat_fee   = {"WEBSITE":0.2,"AMAZON":0.0,"EBAY":0.0}
    platform_code_to_id = {r.platform_code: int(r.platform_id) for r in platforms.itertuples()}

    start_date, end_date = datetime(2024,1,1), datetime(2025,10,15)
    carriers = ["DPD","Royal Mail","Evri"]
    statuses = ["Pending","Shipped","Delivered","Returned","Cancelled"]

    # orders & related
    orders, order_items, order_fees, payments, shipments, returns, return_items = [], [], [], [], [], [], []
    oid=1; oi=1; pay=1; ship=1; ret=1; ri=1

    for _ in range(n_orders):
        plat_code = rchoice_w(["AMAZON","EBAY","WEBSITE"],[0.62,0.18,0.20])
        platform_id = platform_code_to_id[plat_code]
        account_id = platform_id
        cust_id = int(np.random.randint(1, n_customers+1))
        dt = rand_date(start_date, end_date)  # canonical, guaranteed
        n_lines = random.choice([1,1,2,2,3])

        subtotal=0; disc=0; tax=0; ship_amt = money(random.choice([0,2.99,3.99,4.99]))
        this_lines=[]
        for ln in range(1, n_lines+1):
            pid = int(np.random.choice(product_ids))
            base_price = product_disc_price[pid] * (1.02 if plat_code=="AMAZON" else 0.99 if plat_code=="EBAY" else 1.00)
            qty = int(np.random.choice([1,1,1,2,2,3]))
            unit_price = money(base_price + np.random.choice([0,0,0,1]))
            ls = money(unit_price*qty)
            ld = money(ls*np.random.choice([0,0.05,0.1,0]))
            lt = money((ls-ld)*0.2)
            subtotal += ls; disc += ld; tax += lt
            variant_id = int(np.random.choice(prod_variants_map[pid]))
            listing_id = int(np.random.choice(listings_map[(pid, platform_id)]))
            this_lines.append({
                "order_item_id": oi, "order_id": oid, "line_number": ln,
                "product_id": pid, "variant_id": variant_id, "listing_id": listing_id,
                "quantity": qty, "unit_price": unit_price, "line_subtotal": ls,
                "line_discount": ld, "line_tax": lt, "line_total": money(ls-ld+lt),
                "unit_cost": float(product_unit_cost[pid]),
                "margin_amount": money(ls - ld - (float(product_unit_cost[pid])*qty))
            })
            oi += 1

        fee = money((subtotal-disc)*platform_fee_percent[plat_code] + platform_flat_fee.get(plat_code,0.0))
        total = money(subtotal - disc + tax + ship_amt)
        status = rchoice_w(statuses,[0.05,0.10,0.72,0.08,0.05])
        delivery_days = int(np.random.choice([2,3,3,4,5,6])) if status in ("Shipped","Delivered","Returned") else None

        # >>> NEW: include split columns directly, using canonical dt
        orders.append({
            "order_id": oid,
            "order_number": f"{dt.year}-AK-{oid:06d}",
            "order_date": dt,                               # full timestamp
            "order_date_only": dt.date().isoformat(),       # YYYY-MM-DD
            "order_time_only": dt.strftime("%H:%M:%S"),     # HH:MM:SS
            "platform_id": platform_id,
            "account_id": account_id,
            "customer_id": cust_id,
            "currency": "GBP",
            "subtotal_amount": money(subtotal),
            "discount_amount": money(disc),
            "tax_amount": money(tax),
            "shipping_amount": ship_amt,
            "channel_fee_amount": fee,
            "total_amount": total,
            "order_status": status,
            "delivery_days": delivery_days
        })
        order_items += this_lines
        order_fees.append({"order_fee_id": len(order_fees)+1,"order_id": oid,"platform_id": platform_id,"fee_type":"Platform","fee_amount": fee})
        payments.append({"payment_id": pay,"order_id": oid,"payment_method": "Card" if plat_code=="WEBSITE" else rchoice_w(["AmazonPay","PayPal","Card"],[0.6,0.2,0.2]),
                         "provider_txn_id": f"TXN{oid:08d}","amount": total,"status": "Captured" if status in ("Shipped","Delivered","Returned") else "Authorized"})
        pay += 1

        if status in ("Shipped","Delivered","Returned"):
            ship_dt = dt + timedelta(days=int(np.random.choice([0,1,1,2])))
            deliv_dt = ship_dt + timedelta(days=(delivery_days or 3))
            shipments.append({"shipment_id": ship,"order_id": oid,"carrier": random.choice(carriers),
                              "tracking_number": f"TRK{ship:010d}","shipped_at": ship_dt,
                              "delivered_at": deliv_dt if status in ("Delivered","Returned") else None,
                              "delivery_status": "OnTime" if (delivery_days or 3) <= 4 else "Delayed"})
            if status=="Returned" or (status=="Delivered" and random.random()<0.06):
                ret_dt = deliv_dt + timedelta(days=int(np.random.choice([2,3,5,7])))
                returns.append({"return_id": ret,"order_id": oid,"return_number": f"RET-{oid:06d}",
                                "status": random.choice(["Initiated","Received","Refunded"]), "initiated_at": ret_dt})
                one = random.choice(this_lines)
                qty_ret = max(1, int(round(one["quantity"]*random.choice([0.5,1]))))
                return_items.append({"return_item_id": ri,"return_id": ret,"order_item_id": one["order_item_id"],
                                     "quantity_returned": qty_ret,"return_reason": random.choice(["Damaged","Wrong Item","Not as Described","Changed Mind"]),
                                     "refund_amount": money(qty_ret*one["unit_price"])})
                ri += 1; ret += 1
            ship += 1
        oid += 1

    orders_df = pd.DataFrame(orders)
    # ensure dtype for .dt access (should already be datetime)
    orders_df["order_date"] = pd.to_datetime(orders_df["order_date"], errors="coerce")

    order_items_df = pd.DataFrame(order_items)
    order_fees_df = pd.DataFrame(order_fees)
    payments_df = pd.DataFrame(payments)
    shipments_df = pd.DataFrame(shipments)
    returns_df = pd.DataFrame(returns)
    return_items_df = pd.DataFrame(return_items)

    # reviews
    review_rows=[]
    for rid in range(1, n_reviews+1):
        pid = int(np.random.choice(product_ids))
        base_r = products_df.loc[products_df.product_id==pid,"rating"].iloc[0]
        rating = int(np.clip(round(np.random.normal(base_r,0.8)),1,5))
        review_rows.append({
            "review_id": f"R-{rid:06d}", "product_id": pid, "variant_id": None,
            "source_platform": rchoice_w(["AMAZON","EBAY","WEBSITE"],[0.7,0.1,0.2]),
            "user_id": f"U-{np.random.randint(10000,99999)}",
            "user_name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "review_title": random.choice(["Great quality","Value for money","Looks premium","Arrived damaged","Not as described","Perfect for my poster"]),
            "review_content": random.choice(["Excellent build and finish.","Good for the price.","Cracked glass on arrival.","Fits A3 perfectly.","Colour slightly different.","Mount included was useful."]),
            "rating": rating
        })
    reviews_df = pd.DataFrame(review_rows)

    # warehouses & inventory
    warehouses_df = pd.DataFrame([
        {"warehouse_id":1,"warehouse_code":"GLA-DC","warehouse_name":"Glasgow DC","city":"Glasgow","country":"UK"},
        {"warehouse_id":2,"warehouse_code":"BHM-DC","warehouse_name":"Birmingham DC","city":"Birmingham","country":"UK"},
    ])
    inventory_df = pd.DataFrame([{
        "inventory_id": i+1,"variant_id": int(v.variant_id),"warehouse_id": random.choice([1,2]),
        "on_hand_qty": random.randint(0,400),"reserved_qty": random.randint(0,20),
        "reorder_point": random.randint(5,40),"safety_stock": random.randint(5,30)
    } for i, v in enumerate(variants_df.itertuples())])

    # ---------- RAW copies for practice ----------
    def to_raw_products(df):
        raw = df.copy()
        raw["actual_price"] = raw["actual_price"].apply(lambda x: f"£{x}" if random.random()<0.7 else str(x))
        raw["discounted_price"] = raw["discounted_price"].apply(lambda x: f"£{x}" if random.random()<0.7 else str(x))
        cat_map = dict(categories[["category_id","category_name"]].values)
        raw["category_name"] = raw["category_id"].map(cat_map)
        raw["category_name"] = raw["category_name"].apply(lambda s: (" "+s.lower()+" ") if random.random()<0.5 else s.upper())
        mask = np.random.rand(len(raw)) < 0.05
        raw.loc[mask,"rating"] = None
        return raw

    def to_raw_orders(df):
        raw = df.copy()
        fmt_opts = ["%Y-%m-%d %H:%M:%S","%d/%m/%Y %H:%M","%d-%b-%Y","%Y/%m/%d"]
        raw["order_date"] = raw["order_date"].apply(lambda d: d.strftime(random.choice(fmt_opts)))
        for col in ["subtotal_amount","discount_amount","tax_amount","shipping_amount","channel_fee_amount","total_amount"]:
            raw[col] = raw[col].apply(lambda x: f"£{x}" if random.random()<0.3 else str(x))
        return raw

    raw_products_df = to_raw_products(products_df)
    raw_orders_df   = to_raw_orders(orders_df)

    # ---------- CLEAN tables ----------
    # Products: keep original generated + also a cleaned version (example of cleaning)
    def clean_products(raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["actual_price"] = clean_money_series(df["actual_price"])
        df["discounted_price"] = clean_money_series(df["discounted_price"])
        df["discount_percentage"] = (1 - (df["discounted_price"]/df["actual_price"])).clip(lower=0, upper=0.9).round(4)
        if "category_name" in df.columns:
            df["category_name"] = standardize_title(df["category_name"])
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(lower=1, upper=5)
        df["rating"] = df.groupby("category_id")["rating"].transform(lambda s: s.fillna(s.mean().round(2) if s.notna().any() else 4.2))
        return df

    clean_products_df = clean_products(raw_products_df)

    # Orders: DO NOT re-parse; derive splits from canonical timestamps
    orders_df["order_date_only"] = orders_df["order_date"].dt.date.astype(str)
    orders_df["order_time_only"] = orders_df["order_date"].dt.strftime("%H:%M:%S")

    # ---------- WRITE CSVs ----------
    def write_csv(df, name, folder):
        df.to_csv(os.path.join(folder, f"{name}.csv"), index=False)

    # clean (canonical)
    write_csv(brands, "brands", clean_dir)
    write_csv(platforms, "platforms", clean_dir)
    write_csv(categories, "categories", clean_dir)
    write_csv(products_df, "products_clean_generated", clean_dir)
    write_csv(clean_products_df, "products", clean_dir)
    write_csv(variants_df, "product_variants", clean_dir)
    write_csv(listings_df, "product_listings", clean_dir)
    write_csv(listing_prices_df, "listing_prices", clean_dir)
    write_csv(platform_fees, "platform_fees", clean_dir)
    write_csv(channel_inventory_df, "channel_inventory", clean_dir)
    write_csv(marketplace_accounts, "marketplace_accounts", clean_dir)
    write_csv(customers_df, "customers", clean_dir)
    write_csv(order_items_df, "order_items", clean_dir)
    write_csv(order_fees_df, "order_fees", clean_dir)
    write_csv(payments_df, "payments", clean_dir)
    write_csv(shipments_df, "shipments", clean_dir)
    write_csv(returns_df, "returns", clean_dir)
    write_csv(return_items_df, "return_items", clean_dir)
    write_csv(reviews_df, "reviews", clean_dir)
    write_csv(warehouses_df, "warehouses", clean_dir)
    write_csv(inventory_df, "inventory", clean_dir)

    # orders (export canonical & split columns)
    write_csv(orders_df, "orders_clean_generated", clean_dir)  # for comparison/debug
    write_csv(orders_df, "orders", clean_dir)                  # main clean file used by loader

    # RAW sources (for cleaning practice)
    raw_products_df.to_csv(os.path.join(raw_dir, "products_raw.csv"), index=False)
    raw_orders_df.to_csv(os.path.join(raw_dir, "orders_raw.csv"), index=False)

    # ---------- DDL (includes split columns) ----------
    ddl = """
CREATE SCHEMA IF NOT EXISTS ag_oltp;

CREATE TABLE IF NOT EXISTS ag_oltp.brands (
  brand_id INT PRIMARY KEY, brand_name TEXT NOT NULL, website_url TEXT
);

CREATE TABLE IF NOT EXISTS ag_oltp.platforms (
  platform_id INT PRIMARY KEY, platform_code TEXT NOT NULL, platform_name TEXT NOT NULL, region TEXT
);

CREATE TABLE IF NOT EXISTS ag_oltp.categories (
  category_id INT PRIMARY KEY, parent_category_id INT NULL, category_name TEXT NOT NULL, category_path TEXT
);

CREATE TABLE IF NOT EXISTS ag_oltp.products (
  product_id INT PRIMARY KEY, sku TEXT, product_name TEXT NOT NULL, category_id INT, brand_id INT,
  actual_price NUMERIC(10,2), discounted_price NUMERIC(10,2), discount_percentage NUMERIC(6,4),
  rating NUMERIC(3,2), rating_count INT, about_product TEXT, img_link TEXT, product_link TEXT,
  status TEXT, unit_cost NUMERIC(10,2), default_list_price NUMERIC(10,2)
);

CREATE TABLE IF NOT EXISTS ag_oltp.product_variants (
  variant_id INT PRIMARY KEY, product_id INT, variant_sku TEXT, size_code TEXT,
  width_mm INT, height_mm INT, frame_material TEXT, frame_finish TEXT, frame_profile TEXT, glazing_type TEXT,
  mount_included_flag BOOLEAN, mount_color TEXT, backing_type TEXT, orientation TEXT,
  unit_cost NUMERIC(10,2), default_list_price NUMERIC(10,2), weight_kg NUMERIC(8,2),
  package_length_mm INT, package_width_mm INT, package_height_mm INT, status TEXT
);

CREATE TABLE IF NOT EXISTS ag_oltp.marketplace_accounts (
  account_id INT PRIMARY KEY, platform_id INT, merchant_slug TEXT, default_currency TEXT
);

CREATE TABLE IF NOT EXISTS ag_oltp.product_listings (
  listing_id INT PRIMARY KEY, product_id INT, variant_id INT NULL, platform_id INT, account_id INT,
  listing_sku TEXT, title TEXT, subtitle TEXT, description_html TEXT, bullets_json JSON,
  main_image_url TEXT, additional_images_json JSON,
  amazon_asin TEXT, amazon_marketplace_id TEXT, amazon_fulfilment_channel TEXT,
  ebay_item_id TEXT, ebay_listing_type TEXT, ebay_condition_id INT, ebay_category_id INT, is_active BOOLEAN
);

CREATE TABLE IF NOT EXISTS ag_oltp.listing_prices (
  price_id INT PRIMARY KEY, listing_id INT, currency TEXT, listing_price NUMERIC(10,2),
  sale_price NUMERIC(10,2), valid_from DATE, valid_to DATE NULL
);

CREATE TABLE IF NOT EXISTS ag_oltp.platform_fees (
  platform_fee_id INT PRIMARY KEY, platform_id INT, fee_type TEXT, fee_percent NUMERIC(6,4), fee_flat_amount NUMERIC(10,2)
);

CREATE TABLE IF NOT EXISTS ag_oltp.channel_inventory (
  channel_inventory_id INT PRIMARY KEY, listing_id INT, on_hand_qty INT, reserved_qty INT, backorder_qty INT
);

CREATE TABLE IF NOT EXISTS ag_oltp.customers (
  customer_id INT PRIMARY KEY, first_name TEXT, last_name TEXT, email TEXT, phone TEXT, gender TEXT, age_group TEXT, region TEXT,
  signup_source TEXT, preferred_platform TEXT, repeat_customer_flag BOOLEAN
);

CREATE TABLE IF NOT EXISTS ag_oltp.orders (
  order_id INT PRIMARY KEY, order_number TEXT, order_date TIMESTAMP,
  order_date_only DATE, order_time_only TIME,
  platform_id INT, account_id INT, customer_id INT, currency TEXT,
  subtotal_amount NUMERIC(12,2), discount_amount NUMERIC(12,2), tax_amount NUMERIC(12,2), shipping_amount NUMERIC(12,2),
  channel_fee_amount NUMERIC(12,2), total_amount NUMERIC(12,2), order_status TEXT, delivery_days INT NULL
);

CREATE TABLE IF NOT EXISTS ag_oltp.order_items (
  order_item_id INT PRIMARY KEY, order_id INT, line_number INT, product_id INT, variant_id INT, listing_id INT,
  quantity INT, unit_price NUMERIC(12,2), line_subtotal NUMERIC(12,2), line_discount NUMERIC(12,2),
  line_tax NUMERIC(12,2), line_total NUMERIC(12,2), unit_cost NUMERIC(12,2), margin_amount NUMERIC(12,2)
);

CREATE TABLE IF NOT EXISTS ag_oltp.order_fees (
  order_fee_id INT PRIMARY KEY, order_id INT, platform_id INT, fee_type TEXT, fee_amount NUMERIC(12,2)
);

CREATE TABLE IF NOT EXISTS ag_oltp.payments (
  payment_id INT PRIMARY KEY, order_id INT, payment_method TEXT, provider_txn_id TEXT, amount NUMERIC(12,2), status TEXT
);

CREATE TABLE IF NOT EXISTS ag_oltp.shipments (
  shipment_id INT PRIMARY KEY, order_id INT, carrier TEXT, tracking_number TEXT, shipped_at TIMESTAMP, delivered_at TIMESTAMP NULL, delivery_status TEXT
);

CREATE TABLE IF NOT EXISTS ag_oltp.returns (
  return_id INT PRIMARY KEY, order_id INT, return_number TEXT, status TEXT, initiated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ag_oltp.return_items (
  return_item_id INT PRIMARY KEY, return_id INT, order_item_id INT, quantity_returned INT, return_reason TEXT, refund_amount NUMERIC(12,2)
);

CREATE TABLE IF NOT EXISTS ag_oltp.reviews (
  review_id TEXT PRIMARY KEY, product_id INT, variant_id INT NULL, source_platform TEXT, user_id TEXT, user_name TEXT, review_title TEXT, review_content TEXT, rating INT
);

CREATE TABLE IF NOT EXISTS ag_oltp.warehouses (
  warehouse_id INT PRIMARY KEY, warehouse_code TEXT, warehouse_name TEXT, city TEXT, country TEXT
);

CREATE TABLE IF NOT EXISTS ag_oltp.inventory (
  inventory_id INT PRIMARY KEY, variant_id INT, warehouse_id INT, on_hand_qty INT, reserved_qty INT, reorder_point INT, safety_stock INT
);
"""
    with open(os.path.join(base_dir, "create_tables.sql"), "w", encoding="utf-8") as f:
        f.write(ddl)

    # optional zip
    if zip_output:
        zip_path = f"{base_dir}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(base_dir, "create_tables.sql"), arcname="create_tables.sql")
            for root in [raw_dir, clean_dir]:
                for fname in os.listdir(root):
                    zf.write(os.path.join(root, fname), arcname=os.path.relpath(os.path.join(root, fname), base_dir))

    # print quick summary
    print({
        "orders": len(orders_df),
        "order_items": len(order_items_df),
        "customers": len(customers_df),
        "products": len(products_df),
        "variants": len(variants_df),
        "reviews": len(reviews_df),
        "base_dir": os.path.abspath(base_dir)
    })

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="./ag_data")
    ap.add_argument("--orders", type=int, default=50000)
    ap.add_argument("--customers", type=int, default=20000)
    ap.add_argument("--products", type=int, default=300)
    ap.add_argument("--reviews", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    generate_dataset(
        base_dir=args.base_dir,
        n_orders=args.orders,
        n_customers=args.customers,
        n_products=args.products,
        n_reviews=args.reviews,
        seed=args.seed,
        zip_output=args.zip
    )

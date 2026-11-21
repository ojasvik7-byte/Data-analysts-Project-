"""
STEP 1 - Create the sample dataset (21_retail_customer_orders.csv)
The project PDF does not ship with data, so we make a realistic one.
Run this ONCE.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)          # fixed seed -> same data every time
n = 500

# Product catalogue: product -> (category, fixed price)
catalogue = {
    "Wireless Mouse":   ("Electronics", 799),
    "Bluetooth Speaker": ("Electronics", 2499),
    "USB-C Cable":      ("Electronics", 299),
    "Headphones":       ("Electronics", 1999),
    "T-Shirt":          ("Clothing", 499),
    "Jeans":            ("Clothing", 1599),
    "Hoodie":           ("Clothing", 1299),
    "Running Shoes":    ("Footwear", 2999),
    "Sandals":          ("Footwear", 699),
    "Notebook Pack":    ("Stationery", 249),
    "Gel Pen Set":      ("Stationery", 149),
    "Desk Lamp":        ("Home", 899),
    "Water Bottle":     ("Home", 399),
    "Coffee Mug":       ("Home", 249),
}
products = rng.choice(list(catalogue), size=n)

df = pd.DataFrame({
    "Order_ID": [f"ORD{1000 + i}" for i in range(n)],
    "Customer_ID": [f"CUST{rng.integers(1, 201):03d}" for _ in range(n)],
    "Order_Date": pd.to_datetime("2025-01-01")
                  + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "Product": products,
    "Category": [catalogue[p][0] for p in products],
    "Quantity": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.45, 0.25, 0.15, 0.1, 0.05]),
    "Unit_Price": [catalogue[p][1] for p in products],
    "Payment_Method": rng.choice(["Credit Card", "Debit Card", "UPI", "Cash on Delivery"],
                                 size=n, p=[0.3, 0.2, 0.35, 0.15]),
})

# A few bulk orders so the outlier step has something to find
bulk = rng.choice(n, 8, replace=False)
df.loc[bulk, "Quantity"] = rng.integers(15, 30, 8)

# Make the data "dirty" on purpose (real data always is)
df.loc[rng.choice(n, 15, replace=False), "Quantity"] = np.nan
df.loc[rng.choice(n, 15, replace=False), "Unit_Price"] = np.nan
dupes = df.sample(10, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)       # 10 duplicate rows

df.to_csv("21_retail_customer_orders.csv", index=False)
print("Created 21_retail_customer_orders.csv with", len(df), "rows")

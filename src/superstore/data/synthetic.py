"""Small synthetic Superstore-like dataset for unit tests and dry runs (NOT for analysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_orders(n: int = 3000, start: str = "2011-01-01", years: int = 4,
                          seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = rng.integers(0, 365 * years, n)
    order_date = pd.Timestamp(start) + pd.to_timedelta(days, unit="D")
    month = order_date.month.to_numpy()
    category = rng.choice(["Furniture", "Office Supplies", "Technology"], n, p=[0.2, 0.6, 0.2])
    base = pd.Series(category).map({"Furniture": 350, "Office Supplies": 120,
                                    "Technology": 450}).to_numpy()
    season = 1 + 0.4 * (month >= 9)  # Q4-ish peak
    discount = rng.choice([0, 0.1, 0.2, 0.3, 0.5], n, p=[0.5, 0.15, 0.2, 0.1, 0.05])
    sales = base * season * rng.lognormal(0, 0.6, n)
    profit = sales * (0.25 - 0.9 * discount) + rng.normal(0, 10, n)
    return pd.DataFrame({
        "Row ID": np.arange(1, n + 1),
        "Order ID": [f"ORD-{i:06d}" for i in range(n)],
        "Order Date": order_date.strftime("%d-%m-%Y"),  # day-first, like Global Superstore
        "Ship Date": (order_date + pd.to_timedelta(rng.integers(0, 7, n), unit="D"))
        .strftime("%d-%m-%Y"),
        "Segment": rng.choice(["Consumer", "Corporate", "Home Office"], n),
        "Market": rng.choice(["US", "EU", "APAC"], n),
        "Category": category,
        "Sales": sales.round(2),
        "Quantity": rng.integers(1, 10, n),
        "Discount": discount,
        "Profit": profit.round(2),
    })

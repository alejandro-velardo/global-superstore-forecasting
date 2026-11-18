"""Data-quality audit and order-level cleaning. All operations are vectorised."""

from __future__ import annotations

import numpy as np
import pandas as pd

SHIP_RANGE = {
    "Same Day": (0, 1), 
    "First Class": (1, 3), 
    "Second Class": (2, 5), 
    "Standard Class": (4, 7)
}

def audit(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing count/%, cardinality and an example value."""
    n = len(df)
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=True),
            "example": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
        }
    ).assign(pct_unique=lambda t: (t["n_unique"] / max(n, 1) * 100).round(2))


def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Boolean mask of values outside [Q1 - k·IQR, Q3 + k·IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)

def validate_orders(o: pd.DataFrame) -> None:
    mode = o["ship_mode"].astype(str)
    lo, hi = mode.map(lambda k: SHIP_RANGE[k][0]), mode.map(lambda k: SHIP_RANGE[k][1])
    assert o["ship_days"].between(lo, hi).all(), "ship_days inconsistent with ship_mode"
    assert not o["flag_ship_before_order"].any(), "ship before order"
    assert o["discount"].between(0, 1).all(), "discount outside [0, 1]"
    assert np.isclose(o["sales"] / o["quantity"], o["unit_price"], rtol=1e-4).all(), "unit_price mismatch"
    assert o.loc[o["postal_code"].notna(), "country"].eq("United States").all(), "postal_code outside US"
    assert o["order_date"].notna().all() and o["sales"].notna().all()

def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Clean order lines and add derived fields.

    Decisions (justify/revise them in the report!):
    - drop exact duplicates and the `row_id` surrogate key;
    - drop rows without order date or sales (cannot be placed in time / no target);
    - never drop outliers or repeated order-product lines: they are flagged
      (flag_outlier_*, flag_dup_key, flag_ship_before_order) because extreme orders are real; 
      decide per model whether to cap them.
    - discount rounded to 2 decimals (float32 noise) before binning;
    - no scaling here to avoid leakage
    """
    log = {"rows_raw": len(df)}     # we will log what each step removes

    out = df.drop(columns=["row_id"], errors="ignore").drop_duplicates()
    log["dropped_exact_dups"] = log["rows_raw"] - len(out)

    # Flag repeated order-product combinations
    if {"order_id", "product_id"} <= set(out.columns):
        out["flag_dup_key"] = out.duplicated(
            ["order_id", "product_id"], keep=False
        )

    n = len(out)
    out = out.dropna(subset=[c for c in ("order_date", "sales") if c in out.columns])
    log["dropped_missing_key"] = n - len(out)

    # a reliable order key
    out["order_key"] = (out["order_id"].astype(str) + "|" + out["customer_id"].astype(str)
                        + "|" + out["order_date"].dt.strftime("%Y-%m-%d"))
    g = out.groupby("order_id", observed=True)
    out["flag_order_id_collision"] = (g["customer_id"].transform("nunique") > 1) | \
                                    (g["order_date"].transform("nunique") > 1)

    # a reliable product key
    out["product_key"] = out["product_id"].astype(str) + "|" + out["product_name"].astype(str)

    if {"order_date", "ship_date"} <= set(out.columns):
        out["ship_days"] = (out["ship_date"] - out["order_date"]).dt.days
        out["flag_ship_before_order"] = out["ship_days"] < 0

    if {"profit", "sales"} <= set(out.columns):
        out["profit_margin"] = np.where(out["sales"] != 0, out["profit"] / out["sales"], np.nan)
        out["is_loss"] = out["profit"] < 0

    if {"sales", "quantity"} <= set(out.columns):
        out["unit_price"] = out["sales"] / out["quantity"].replace(0, np.nan)

    if "discount" in out.columns:
        out["discount"] = out["discount"].astype("float64").round(2)   # also removes 0.002 / 0.202 / 0.402 to prevent errors in the band edges
        out["discount_band"] = pd.cut(
            out["discount"],
            bins=[-np.inf, 0, 0.1, 0.2, 0.3, 0.5, np.inf],
            labels=["0", "(0,10]", "(10,20]", "(20,30]", "(30,50]", ">50"],
        )

    for col in ("sales", "profit"):
        if col in out.columns:
            out[f"flag_outlier_{col}"] = iqr_outlier_mask(out[col])

    out = out.sort_values("order_date", kind="stable").reset_index(drop=True)
    log["rows_clean"] = len(out)
    out.attrs["cleaning_log"] = log
    validate_orders(out)

    return out

"""Data-quality audit and order-level cleaning. All operations are vectorised."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Clean order lines and add derived fields.

    Decisions (justify/revise them in the report!):
    - drop exact duplicates and the `row_id` surrogate key;
    - drop rows without order date or sales (cannot be placed in time / no target);
    - flag (do NOT drop) rows with ship_date < order_date and IQR outliers — retail extremes are
      usually real large orders, not errors; decide per model whether to cap them.
    """
    out = df.drop(columns=["row_id"], errors="ignore").drop_duplicates()
    out = out.dropna(subset=[c for c in ("order_date", "sales") if c in out.columns])

    if {"order_date", "ship_date"} <= set(out.columns):
        out["ship_days"] = (out["ship_date"] - out["order_date"]).dt.days
        out["flag_ship_before_order"] = out["ship_days"] < 0

    if {"profit", "sales"} <= set(out.columns):
        out["profit_margin"] = np.where(out["sales"] != 0, out["profit"] / out["sales"], np.nan)
        out["is_loss"] = out["profit"] < 0

    if {"sales", "quantity"} <= set(out.columns):
        out["unit_price"] = out["sales"] / out["quantity"].replace(0, np.nan)

    if "discount" in out.columns:
        out["discount_band"] = pd.cut(
            out["discount"],
            bins=[-np.inf, 0, 0.1, 0.2, 0.3, 0.5, np.inf],
            labels=["0", "(0,10]", "(10,20]", "(20,30]", "(30,50]", ">50"],
        )

    for col in ("sales", "profit"):
        if col in out.columns:
            out[f"flag_outlier_{col}"] = iqr_outlier_mask(out[col])

    return out.sort_values("order_date").reset_index(drop=True)

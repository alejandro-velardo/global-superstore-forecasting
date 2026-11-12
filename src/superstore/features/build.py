"""Turn order lines into a regular time-series panel and engineer temporal features.

Leakage rule: every lag / rolling feature for period t only uses data up to t-1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def to_period_start(dates: pd.Series, freq: str) -> pd.Series:
    """Map timestamps to the start of their period ("MS" monthly, "W-MON" weekly, "D" daily)."""
    if freq == "MS":
        return dates.dt.to_period("M").dt.start_time
    if freq == "QS":
        return dates.dt.to_period("Q").dt.start_time
    if freq == "W-MON":
        return dates.dt.to_period("W-SUN").dt.start_time  # weeks Mon..Sun, labelled by Monday
    if freq == "D":
        return dates.dt.normalize()
    raise ValueError(f"Unsupported freq {freq!r}; use 'D', 'W-MON', 'MS' or 'QS'.")


def aggregate_panel(
    orders: pd.DataFrame,
    freq: str = "MS",
    group_cols: Sequence[str] = ("category",),
    date_col: str = "order_date",
) -> pd.DataFrame:
    """Aggregate order lines to (group, period) with a complete, gap-free date grid.

    Periods with no orders are filled with 0 sales/profit/quantity (true zeros, not missing).
    Discount is averaged weighted by sales.
    """
    group_cols = list(group_cols)
    df = orders.assign(
        period=to_period_start(orders[date_col], freq),
        _disc_x_sales=orders.get("discount", 0) * orders["sales"],
    )
    agg_spec = {"sales": ("sales", "sum"), "n_lines": ("sales", "size")}
    for col in ("profit", "quantity"):
        if col in df.columns:
            agg_spec[col] = (col, "sum")
    if "discount" in df.columns:
        agg_spec["_disc_x_sales"] = ("_disc_x_sales", "sum")
        agg_spec["discount_mean"] = ("discount", "mean")

    panel = df.groupby([*group_cols, "period"], observed=True).agg(**agg_spec)

    # Complete grid: every group × every period
    periods = pd.date_range(panel.index.get_level_values("period").min(),
                            panel.index.get_level_values("period").max(), freq=freq)
    groups = panel.index.droplevel("period").unique()
    full_idx = pd.MultiIndex.from_tuples(
        [(*(g if isinstance(g, tuple) else (g,)), p) for g in groups for p in periods],
        names=[*group_cols, "period"],
    )
    panel = panel.reindex(full_idx).fillna(0).reset_index()

    if "_disc_x_sales" in panel.columns:
        panel["discount_wavg"] = np.where(
            panel["sales"] > 0, panel["_disc_x_sales"] / panel["sales"], 0.0
        )
        panel = panel.drop(columns="_disc_x_sales")
    if "profit" in panel.columns:
        panel["profit_margin"] = np.where(panel["sales"] > 0, panel["profit"] / panel["sales"], 0.0)
    return panel


def add_calendar_features(df: pd.DataFrame, date_col: str = "period") -> pd.DataFrame:
    """Calendar features + cyclical (sin/cos) month encoding."""
    d = df[date_col].dt
    month = d.month
    return df.assign(
        year=d.year,
        quarter=d.quarter,
        month=month,
        month_sin=np.sin(2 * np.pi * month / 12),
        month_cos=np.cos(2 * np.pi * month / 12),
        is_q4=(d.quarter == 4).astype("int8"),
        t=(d.year - d.year.min()) * 12 + month,  # linear trend index
    )


def add_lag_features(
    df: pd.DataFrame,
    target: str = "sales",
    group_cols: Sequence[str] = ("category",),
    lags: Sequence[int] = (1, 2, 3, 12),
    windows: Sequence[int] = (3, 6, 12),
    date_col: str = "period",
) -> pd.DataFrame:
    """Per-group lags and rolling stats of `target`, computed on shifted data (no leakage)."""
    group_cols = list(group_cols)
    out = df.sort_values([*group_cols, date_col]).copy()
    g = out.groupby(group_cols, observed=True)[target]
    for lag in lags:
        out[f"{target}_lag{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    keys = [out[c] for c in group_cols]
    for w in windows:
        roll = shifted.groupby(keys, observed=True).rolling(w, min_periods=1)
        # groupby().rolling() prepends the group keys to the index -> drop them to realign
        out[f"{target}_rmean{w}"] = roll.mean().droplevel(list(range(len(group_cols))))
        out[f"{target}_rstd{w}"] = roll.std().droplevel(list(range(len(group_cols))))
    return out


def add_discount_interactions(
    df: pd.DataFrame, discount_col: str = "discount_wavg", by: str = "category"
) -> pd.DataFrame:
    """discount × group dummies and discount² — lets linear models learn group-specific,
    non-linear discount effects (central to the 'discount impact' research question)."""
    if discount_col not in df.columns or by not in df.columns:
        return df
    dummies = pd.get_dummies(df[by], prefix=f"disc_x_{by}", dtype=float)
    return pd.concat([df, dummies.mul(df[discount_col], axis=0)], axis=1).assign(
        **{f"{discount_col}_sq": df[discount_col] ** 2}
    )


def build_panel(
    orders: pd.DataFrame,
    freq: str = "MS",
    group_cols: Sequence[str] = ("category",),
    target: str = "sales",
    lags: Sequence[int] = (1, 2, 3, 12),
    windows: Sequence[int] = (3, 6, 12),
) -> pd.DataFrame:
    """Full feature pipeline: aggregate → calendar → lags/rolling → discount interactions."""
    panel = aggregate_panel(orders, freq=freq, group_cols=group_cols)
    panel = add_calendar_features(panel)
    panel = add_lag_features(panel, target=target, group_cols=group_cols, lags=lags,
                             windows=windows)
    # Lag the discount too: at forecast time we know *planned* discount only if the business sets
    # it in advance — keep both and discuss the assumption in the report.
    if "discount_wavg" in panel.columns:
        panel["discount_wavg_lag1"] = panel.groupby(list(group_cols), observed=True)[
            "discount_wavg"
        ].shift(1)
    panel = add_discount_interactions(panel, by=group_cols[0])
    return panel.reset_index(drop=True)

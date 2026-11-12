import numpy as np
import pandas as pd

from superstore.features.build import add_lag_features, aggregate_panel, build_panel


def test_panel_is_complete_grid(orders):
    panel = aggregate_panel(orders, freq="MS", group_cols=["category"])
    n_groups = orders["category"].nunique()
    n_periods = panel["period"].nunique()
    assert len(panel) == n_groups * n_periods
    assert not panel.duplicated(["category", "period"]).any()


def test_panel_preserves_totals(orders):
    panel = aggregate_panel(orders, freq="MS", group_cols=["category"])
    assert np.isclose(panel["sales"].sum(), orders["sales"].sum())


def test_lags_have_no_leakage():
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5,
                       "period": list(pd.date_range("2020-01-01", periods=5, freq="MS")) * 2,
                       "sales": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]})
    out = add_lag_features(df, group_cols=["g"], lags=[1], windows=[2]).set_index(["g", "period"])
    a = out.loc["a"]
    assert np.isnan(a["sales_lag1"].iloc[0])           # nothing before the first period
    assert a["sales_lag1"].iloc[3] == 3                 # t=4 sees t=3
    assert a["sales_rmean2"].iloc[3] == 2.5             # mean(2, 3), excludes current value 4
    assert out.loc["b", "sales_lag1"].iloc[1] == 10     # groups don't bleed into each other


def test_build_panel_weekly(orders):
    panel = build_panel(orders, freq="W-MON", group_cols=["market", "category"], lags=[1, 52])
    assert (panel["period"].dt.dayofweek == 0).all()
    assert "sales_lag52" in panel.columns

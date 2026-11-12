import numpy as np
import pandas as pd
import pytest

from superstore.evaluation.cv import PanelTimeSeriesSplit
from superstore.evaluation.metrics import mase, wape
from superstore.features.build import build_panel
from superstore.models.train import run_cv, select_features


def test_cv_is_strictly_temporal():
    dates = pd.Series(np.repeat(pd.date_range("2020-01-01", periods=24, freq="MS"), 3))
    cv = PanelTimeSeriesSplit(n_splits=3, horizon=2, gap=1, min_train=12)
    folds = list(cv.split(dates))
    assert len(folds) == 3
    for tr, te in folds:
        assert dates.iloc[tr].max() < dates.iloc[te].min()
        assert dates.iloc[te].nunique() == 2
    assert folds[-1][1].max() == len(dates) - 1  # last fold ends at the last period


def test_cv_raises_if_too_short():
    dates = pd.Series(pd.date_range("2020-01-01", periods=10, freq="MS"))
    with pytest.raises(ValueError):
        list(PanelTimeSeriesSplit(n_splits=3, horizon=3, min_train=12).split(dates))


def test_metrics():
    assert wape([10, 10], [10, 10]) == 0
    assert wape([10, 10], [5, 15]) == 0.5
    assert mase([2, 3], [2, 3], y_train=[1, 2, 3, 4]) == 0


def test_no_target_or_same_period_leaks_in_features(orders):
    panel = build_panel(orders, group_cols=["category"])
    nums, cats = select_features(panel)
    forbidden = {"sales", "profit", "quantity", "discount_wavg", "profit_margin", "n_lines"}
    assert not forbidden & set(nums + cats)


def test_run_cv_end_to_end(orders):
    panel = build_panel(orders, group_cols=["category"])
    res = run_cv(panel, cv=PanelTimeSeriesSplit(n_splits=2, horizon=3, min_train=13))
    assert res["wape"].notna().all()
    assert {"naive (last period)", "ridge (log)", "hist gradient boosting"} <= set(res["model"])

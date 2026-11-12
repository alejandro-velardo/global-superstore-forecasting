"""Candidate models and the cross-validated experiment runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from superstore.config import RANDOM_STATE
from superstore.evaluation.cv import PanelTimeSeriesSplit
from superstore.evaluation.metrics import all_metrics
from superstore.models.baselines import baseline_models

NON_FEATURES = {"period", "sales", "profit", "quantity", "n_lines", "profit_margin",
                "discount_mean", "discount_wavg"}


def select_features(panel: pd.DataFrame, target: str = "sales") -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature columns, excluding the target and same-period
    quantities that would not be known at forecast time."""
    exclude = NON_FEATURES | {target}
    disc_now = [c for c in panel.columns if c.startswith("disc_x_") or c == "discount_wavg_sq"]
    exclude |= set(disc_now)  # same-period discount: enable only if discount is planned ahead
    cats = [c for c in panel.select_dtypes(["object", "category", "string"]).columns
            if c not in exclude]
    nums = [c for c in panel.select_dtypes("number").columns if c not in exclude]
    return nums, cats


def _preprocessor(nums: Sequence[str], cats: Sequence[str], scale: bool) -> ColumnTransformer:
    num_steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    return ColumnTransformer(
        [("num", make_pipeline(*num_steps), list(nums)),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cats))]
    )


def candidate_models(nums: Sequence[str], cats: Sequence[str]) -> dict[str, BaseEstimator]:
    """Parametric (regularised linear) vs. non-parametric (tree ensembles).

    Linear models are fit on log1p(sales) — sales are right-skewed and multiplicative effects
    (seasonality × level) become additive in log space.
    """
    def log_target(est):
        return TransformedTargetRegressor(est, func=np.log1p, inverse_func=np.expm1)

    alphas = np.logspace(-3, 3, 25)
    models = {
        "ridge (log)": log_target(make_pipeline(_preprocessor(nums, cats, True),
                                                RidgeCV(alphas=alphas))),
        "lasso (log)": log_target(make_pipeline(_preprocessor(nums, cats, True),
                                                LassoCV(alphas=alphas, max_iter=20_000,
                                                        random_state=RANDOM_STATE))),
        "random forest": make_pipeline(_preprocessor(nums, cats, False),
                                       RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                             random_state=RANDOM_STATE, n_jobs=-1)),
        "hist gradient boosting": make_pipeline(
            _preprocessor(nums, cats, False),
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          random_state=RANDOM_STATE)),
    }
    try:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = make_pipeline(
            _preprocessor(nums, cats, False),
            LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                          min_child_samples=5, random_state=RANDOM_STATE, verbose=-1))
    except ImportError:
        pass
    return models


def run_cv(
    panel: pd.DataFrame,
    models: Mapping[str, BaseEstimator] | None = None,
    target: str = "sales",
    cv: PanelTimeSeriesSplit | None = None,
    season: int = 12,
    date_col: str = "period",
) -> pd.DataFrame:
    """Evaluate baselines + candidate models with the same temporal folds.

    Returns one row per (model, fold) with all metrics; aggregate with
    `res.groupby("model")[metrics].mean()`.
    """
    nums, cats = select_features(panel, target)
    if models is None:
        models = {**baseline_models(target, season), **candidate_models(nums, cats)}
    cv = cv or PanelTimeSeriesSplit(n_splits=3, horizon=3, min_train=season + 1)

    X, y = panel[nums + cats], panel[target].to_numpy(float)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(panel[date_col]), start=1):
        for name, model in models.items():
            est = clone(model).fit(X.iloc[tr], y[tr])
            pred = np.clip(est.predict(X.iloc[te]), 0, None)  # sales cannot be negative
            rows.append({"model": name, "fold": fold,
                         "test_start": panel[date_col].iloc[te].min(), **all_metrics(y[te], pred)})
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, sort_by: str = "wape") -> pd.DataFrame:
    metric_cols = ["mae", "rmse", "wape", "smape", "r2"]
    return (results.groupby("model")[metric_cols].agg(["mean", "std"])
            .sort_values((sort_by, "mean")).round(4))

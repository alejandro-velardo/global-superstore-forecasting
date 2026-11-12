"""Forecasting baselines. Any model worth reporting must beat these.

They read the pre-computed (leak-free) lag columns, so they plug into the same CV loop as ML models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class ColumnBaseline(BaseEstimator, RegressorMixin):
    """Predict ŷ = X[column], falling back to `fallback` column, then to the training mean."""

    def __init__(self, column: str, fallback: str | None = None):
        self.column = column
        self.fallback = fallback

    def fit(self, X: pd.DataFrame, y=None):
        self.mean_ = float(np.nanmean(y)) if y is not None else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = X[self.column]
        if self.fallback is not None:
            pred = pred.fillna(X[self.fallback])
        return pred.fillna(self.mean_).to_numpy(float)


def baseline_models(target: str = "sales", season: int = 12) -> dict[str, ColumnBaseline]:
    return {
        "naive (last period)": ColumnBaseline(f"{target}_lag1"),
        f"seasonal naive (t-{season})": ColumnBaseline(f"{target}_lag{season}",
                                                       fallback=f"{target}_lag1"),
        "moving average (3)": ColumnBaseline(f"{target}_rmean3"),
    }

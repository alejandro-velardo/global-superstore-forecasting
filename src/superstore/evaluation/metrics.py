"""Forecast error metrics. Prefer scale-free metrics (WAPE, MASE) to compare across groups."""

from __future__ import annotations

import numpy as np


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def wape(y, yhat) -> float:
    """Weighted absolute percentage error = Σ|e| / Σ|y|.

    Robust to zero-sales periods, unlike MAPE."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom > 0 else np.nan


def smape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y) + np.abs(yhat)
    ratio = np.divide(2 * np.abs(y - yhat), denom, out=np.zeros_like(denom), where=denom > 0)
    return float(ratio.mean())


def mase(y, yhat, y_train, season: int = 1) -> float:
    """Mean absolute scaled error vs. in-sample seasonal-naive error (<1 beats seasonal naive)."""
    y_train = np.asarray(y_train, float)
    if len(y_train) <= season:
        return np.nan
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return float(mae(y, yhat) / scale) if scale and scale > 0 else np.nan


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan


def all_metrics(y, yhat) -> dict[str, float]:
    return {"mae": mae(y, yhat), "rmse": rmse(y, yhat), "wape": wape(y, yhat),
            "smape": smape(y, yhat), "r2": r2(y, yhat)}

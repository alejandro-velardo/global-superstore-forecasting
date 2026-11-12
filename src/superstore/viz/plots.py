"""Shared plotting style and reusable figures, so every chart in the report looks consistent."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from superstore.config import FIGURES_DIR


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False})


def save_fig(fig: plt.Figure, name: str, fmt: str = "pdf") -> None:
    """Save to reports/figures/<name>.<fmt> (vector PDF by default — crisp in LaTeX)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.{fmt}")


def plot_series(panel: pd.DataFrame, y: str = "sales", hue: str = "category",
                date_col: str = "period", ax=None) -> plt.Axes:
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    sns.lineplot(data=panel, x=date_col, y=y, hue=hue, marker="o", markersize=3, ax=ax)
    ax.set(title=f"{y.title()} per {date_col}", xlabel="", ylabel=y)
    return ax


def plot_discount_profit(orders: pd.DataFrame, by: str = "category", ax=None) -> plt.Axes:
    """Mean profit margin per discount band and group — the core 'discount impact' figure."""
    ax = ax or plt.subplots(figsize=(9, 4))[1]
    data = (orders.groupby(["discount_band", by], observed=True)["profit_margin"]
            .mean().reset_index())
    sns.barplot(data=data, x="discount_band", y="profit_margin", hue=by, ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Profit margin by discount band", xlabel="Discount (%)", ylabel="Mean margin")
    return ax


def plot_cv_results(results: pd.DataFrame, metric: str = "wape", ax=None) -> plt.Axes:
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    order = results.groupby("model")[metric].mean().sort_values().index
    sns.pointplot(data=results, y="model", x=metric, order=order, errorbar="sd",
                  linestyle="none", ax=ax)
    ax.set(title=f"Cross-validated {metric.upper()} (mean ± sd over folds)", ylabel="")
    return ax

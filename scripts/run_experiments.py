"""Cross-validated comparison of baselines vs. ML models on the processed panel.

    python scripts/run_experiments.py --horizon 3 --splits 4
"""

import argparse

import pandas as pd

from superstore import config
from superstore.evaluation.cv import PanelTimeSeriesSplit
from superstore.models.train import run_cv, summarise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=config.TARGET)
    p.add_argument("--freq", default=config.PANEL_FREQ, help="must match make_dataset.py")
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--gap", type=int, default=0)
    args = p.parse_args()

    panel = pd.read_parquet(config.PANEL_FILE)
    season = config.SEASON_LENGTH.get(args.freq, 12)
    cv = PanelTimeSeriesSplit(n_splits=args.splits, horizon=args.horizon, gap=args.gap,
                              min_train=season + 1)
    results = run_cv(panel, target=args.target, cv=cv, season=season)

    out = config.REPORTS_DIR / "cv_results.csv"
    results.to_csv(out, index=False)
    print(summarise(results).to_string())
    print("\nPer-fold results →", out)


if __name__ == "__main__":
    main()

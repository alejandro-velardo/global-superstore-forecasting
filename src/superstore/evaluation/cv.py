"""Time-series cross-validation for panel data.

Random K-fold leaks the future into training. Instead we use an expanding window over *periods*:
each fold trains on all periods < cutoff and tests on the next `horizon` periods, for all groups
at once. An optional `gap` simulates the delay between the last observed data and the forecast.

    fold 1: [train ........][gap][test]
    fold 2: [train ...............][gap][test]
    fold 3: [train ......................][gap][test]
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


class PanelTimeSeriesSplit:
    def __init__(self, n_splits: int = 3, horizon: int = 3, gap: int = 0, min_train: int = 12):
        self.n_splits, self.horizon, self.gap, self.min_train = n_splits, horizon, gap, min_train

    def split(self, dates: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        periods = np.sort(pd.unique(dates))
        n = len(periods)
        needed = self.min_train + self.gap + self.n_splits * self.horizon
        if n < needed:
            raise ValueError(f"Need ≥{needed} periods for this CV setup, got {n}.")
        values = dates.to_numpy()
        for k in range(self.n_splits, 0, -1):
            test_end = n - (k - 1) * self.horizon
            test_start = test_end - self.horizon
            train_end = test_start - self.gap
            train_mask = values < periods[train_end]
            # train_end index is exclusive; periods[train_end] may be in the gap/test
            test_mask = (values >= periods[test_start]) & (values <= periods[test_end - 1])
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)

    def get_n_splits(self, *_, **__) -> int:
        return self.n_splits

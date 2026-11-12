# Global Superstore — Retail Analytics & Profitability Forecasting

UAB · Master in Modelling for Science and Engineering · Python Module · Applied Data Science Project (2026–2027)
Topic #18 — **Time-Series / Supervised Regression** · Deadline: **9 December 2026, 23:59**

> Build an automated data pipeline to parse order dates, calculate profit margins and aggregate discount
> impacts. Construct predictive regression/forecasting models, theoretically analysing discount interaction
> terms and seasonal trends.

## Research questions (draft — refine as a team)

1. **Forecasting:** How well can weekly/monthly sales (and profit) per Market × Category be forecast, and
   do ML models with lag/calendar features beat naive and seasonal-naive baselines?
2. **Discount effects:** Is the effect of discount on profit linear, and does it differ by Category /
   Sub-Category / Region (interaction terms)? Is there a discount threshold where orders become loss-making?
3. **Seasonality:** Which seasonal patterns (month, quarter, Q4 peak) drive demand, and are they stable
   across markets?

## Repository layout

```
.
├── data/
│   ├── raw/          # original Kaggle files (git-ignored, never edited)
│   ├── interim/      # cleaned order-level data (parquet)
│   └── processed/    # model-ready panels (parquet)
├── notebooks/        # numbered, narrative notebooks — import logic from src/, keep them thin
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modelling.ipynb
├── src/superstore/   # importable package: all reusable, tested logic lives here
│   ├── config.py           # paths, constants, random seed
│   ├── data/load.py        # robust CSV/XLSX loader, column normalisation, date parsing
│   ├── data/clean.py       # audit, cleaning, derived fields (margin, ship days, outlier flags)
│   ├── features/build.py   # time-series aggregation, calendar, lag/rolling, discount interactions
│   ├── evaluation/cv.py    # expanding-window time-series cross-validation
│   ├── evaluation/metrics.py
│   ├── models/baselines.py # naive, seasonal-naive, moving-average
│   ├── models/train.py     # candidate models + CV experiment runner
│   └── viz/plots.py        # shared plotting style and helpers
├── scripts/          # CLI entry points (download → make dataset → run experiments)
├── tests/            # pytest unit tests on synthetic data
├── models/           # serialized models (git-ignored)
├── reports/
│   ├── figures/      # exported figures used in the report
│   └── manuscript/   # LaTeX report (final PDF goes here)
└── docs/
    ├── ai_usage_log.md     # running log for the mandatory GenAI declaration
    └── contributions.md    # who did what (required report section)
```

## Setup

Requires Python ≥ 3.10.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # also installs this repo as an editable package (src/superstore)
```

Or simply `make setup`.

## Data

Download the Superstore dataset from Kaggle (the link in the assignment PDF) and put the CSV/XLSX in
`data/raw/`. Either do it manually or, with a Kaggle API token in `~/.kaggle/kaggle.json`:

```bash
python scripts/download_data.py --dataset <owner>/<dataset-slug>
```

The loader auto-detects the file, encoding (UTF-8 / Latin-1) and date format (day-first vs month-first),
and normalises column names to `snake_case`, so both the *Sample Superstore* and the *Global Superstore*
variants work. See `data/README.md`.

## Pipeline

```bash
make data      # raw → data/interim/orders.parquet → data/processed/panel.parquet
make train     # baselines vs ML models with time-series CV → reports/cv_results.csv
make test      # unit tests
make lint      # ruff
```

Every step is also callable from Python, e.g.

```python
from superstore.data.load import load_raw
from superstore.data.clean import clean_orders
from superstore.features.build import build_panel

orders = clean_orders(load_raw())
panel = build_panel(orders, freq="MS", group_cols=["market", "category"], target="sales")
```

## Conventions

- **Logic in `src/`, narrative in notebooks.** If a cell is copied twice, move it into a function.
- **Vectorised pandas/numpy only** — no row-wise `apply`/loops over rows (it's graded).
- **No leakage:** lags/rolling windows are always built on shifted data; splits are strictly temporal.
- Fixed seed in `superstore.config.RANDOM_STATE`.
- Branch per task (`feat/eda-discounts`, `feat/lgbm`), small PRs, one reviewer.
- Log every meaningful GenAI interaction in `docs/ai_usage_log.md` *as you go*.
- Save report figures with `superstore.viz.plots.save_fig(fig, "name")` → `reports/figures/`.

## Team

| Member | Main role |
|---|---|
| Alejandro Velardo | _TBD_ |
| _Member 2_ | _TBD_ |
| _Member 3_ | _TBD_ |
| _Member 4_ | _TBD_ |

Instructor access: add **victor.navas@uab.cat** (or his GitHub/GitLab user) to the private repository.

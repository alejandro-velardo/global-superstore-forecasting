"""Project-wide paths and constants. Import from here instead of hard-coding."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

ORDERS_FILE = INTERIM_DIR / "orders.parquet"
PANEL_FILE = PROCESSED_DIR / "panel.parquet"

RANDOM_STATE = 42

DATE_COLS = ("order_date", "ship_date")
TARGET = "sales"

# Default aggregation for the forecasting panel
PANEL_FREQ = "MS"  # month start; use "W-MON" for weekly
PANEL_GROUPS = ("category",)  # add "market"/"region" for a richer panel
SEASON_LENGTH = {"MS": 12, "W-MON": 52, "D": 7}

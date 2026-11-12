"""Robust loading of the raw Superstore file(s).

Handles the different Kaggle variants (Sample Superstore / Global Superstore):
- CSV or Excel, UTF-8 or Latin-1 encoding
- column names normalised to snake_case ("Sub-Category" -> "sub_category")
- day-first vs month-first dates detected from the data itself
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from superstore.config import DATE_COLS, RAW_DIR

_ENCODINGS = ("utf-8", "latin-1")


def to_snake(name: str) -> str:
    """'Sub-Category' -> 'sub_category', 'Row ID' -> 'row_id'."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip())
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return name.strip("_").lower()


def find_raw_file(raw_dir: Path = RAW_DIR) -> Path:
    """Return the single data file in data/raw (largest one if several)."""
    candidates = [
        p for p in raw_dir.iterdir() if p.suffix.lower() in {".csv", ".xlsx", ".xls"}
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No CSV/XLSX found in {raw_dir}. Download the dataset first (see README)."
        )
    return max(candidates, key=lambda p: p.stat().st_size)


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    last_err: Exception | None = None
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as err:
            last_err = err
    raise ValueError(f"Could not decode {path} with {_ENCODINGS}") from last_err


def parse_dates(df: pd.DataFrame, cols=DATE_COLS) -> pd.DataFrame:
    """Parse date columns, choosing day-first or month-first per dataset.

    Strings like '03-04-2014' are ambiguous. We parse with both conventions and keep the one
    that (1) yields fewer NaT and (2) yields more rows with ship_date >= order_date.
    No `format="mixed"` on purpose: it re-guesses per element and would silently swap day and
    month on some rows. A single inferred format per column makes wrong conventions fail loudly
    (NaT), which is exactly what the scoring detects.
    """
    cols = [c for c in cols if c in df.columns]
    if not cols or all(pd.api.types.is_datetime64_any_dtype(df[c]) for c in cols):
        return df

    def _parse(dayfirst: bool) -> pd.DataFrame:
        with warnings.catch_warnings():  # pandas warns when the guessed convention mismatches
            warnings.simplefilter("ignore", UserWarning)
            return _parse_raw(dayfirst)

    def _parse_raw(dayfirst: bool) -> pd.DataFrame:
        return pd.DataFrame(
            {
                c: pd.to_datetime(df[c], dayfirst=dayfirst, errors="coerce")
                for c in cols
            }
        )

    options = {flag: _parse(flag) for flag in (True, False)}

    def _score(parsed: pd.DataFrame) -> tuple[int, int]:
        n_valid = int(parsed.notna().all(axis=1).sum())
        n_consistent = (
            int((parsed["ship_date"] >= parsed["order_date"]).sum())
            if {"order_date", "ship_date"} <= set(parsed.columns)
            else 0
        )
        return n_valid, n_consistent

    best = max(options, key=lambda flag: _score(options[flag]))
    out = df.copy()
    out[cols] = options[best]
    return out


def optimise_dtypes(df: pd.DataFrame, max_cat_ratio: float = 0.05) -> pd.DataFrame:
    """Downcast numerics and convert low-cardinality strings to `category` (memory + speed)."""
    out = df.copy()
    for col in out.select_dtypes("number").columns:
        kind = "integer" if pd.api.types.is_integer_dtype(out[col]) else "float"
        out[col] = pd.to_numeric(out[col], downcast=kind)
    obj_cols = out.select_dtypes(["object", "string"]).columns
    n = max(len(out), 1)
    for col in obj_cols:
        if out[col].nunique(dropna=True) / n <= max_cat_ratio:
            out[col] = out[col].astype("category")
    return out


def load_raw(path: str | Path | None = None, optimise: bool = True) -> pd.DataFrame:
    """Load the raw dataset with normalised columns and parsed dates."""
    path = Path(path) if path is not None else find_raw_file()
    df = _read_file(path)
    df.columns = [to_snake(c) for c in df.columns]
    # Postal codes are identifiers, not numbers (leading zeros, NaN for non-US rows)
    if "postal_code" in df.columns:
        df["postal_code"] = (
            df["postal_code"].astype("Int64").astype("string")
            if pd.api.types.is_numeric_dtype(df["postal_code"])
            else df["postal_code"].astype("string")
        )
    df = parse_dates(df)
    if optimise:
        df = optimise_dtypes(df)
    return df.replace([np.inf, -np.inf], np.nan)

"""raw → interim/orders.parquet → processed/panel.parquet

    python scripts/make_dataset.py                          # defaults from config.py
    python scripts/make_dataset.py --freq W-MON --groups market category
    python scripts/make_dataset.py --synthetic              # dry run without the real data
"""

import argparse

from superstore import config
from superstore.data.clean import audit, clean_orders
from superstore.data.load import load_raw
from superstore.features.build import build_panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--freq", default=config.PANEL_FREQ)
    p.add_argument("--groups", nargs="+", default=list(config.PANEL_GROUPS))
    p.add_argument("--target", default=config.TARGET)
    p.add_argument("--synthetic", action="store_true", help="use generated data (pipeline test)")
    args = p.parse_args()

    if args.synthetic:
        from superstore.data.load import optimise_dtypes, parse_dates, to_snake
        from superstore.data.synthetic import make_synthetic_orders

        raw = make_synthetic_orders()
        raw.columns = [to_snake(c) for c in raw.columns]
        raw = optimise_dtypes(parse_dates(raw))
    else:
        raw = load_raw()

    print(f"Raw: {raw.shape[0]:,} rows × {raw.shape[1]} cols")
    print(audit(raw).to_string())

    orders = clean_orders(raw)
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(config.ORDERS_FILE, index=False)
    print(f"\nOrders → {config.ORDERS_FILE} ({len(orders):,} rows)")

    season = config.SEASON_LENGTH.get(args.freq, 12)
    panel = build_panel(orders, freq=args.freq, group_cols=args.groups, target=args.target,
                        lags=sorted({1, 2, 3, season}))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.PANEL_FILE, index=False)
    print(f"Panel  → {config.PANEL_FILE} ({panel.shape[0]:,} rows × {panel.shape[1]} cols)")


if __name__ == "__main__":
    main()

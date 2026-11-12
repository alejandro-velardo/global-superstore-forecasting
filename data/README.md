# Data

| Folder | Content | In git? |
|---|---|---|
| `raw/` | Original Kaggle download. **Never edit by hand.** | No |
| `interim/` | `orders.parquet` — cleaned, typed, order-line level | No |
| `processed/` | `panel.parquet` — aggregated time-series panel with features | No |

All intermediate files are regenerated with `make data`.

## Source

Kaggle — Superstore Sales Dataset (link in the assignment PDF, topic #18).
Record here the exact slug, file name, download date and row count once downloaded:

- Slug: `TODO`
- File: `TODO`
- Downloaded: `TODO`
- Rows × columns: `TODO`

## Expected columns (after `snake_case` normalisation)

`order_id, order_date, ship_date, ship_mode, customer_id, customer_name, segment, city, state, country,
postal_code, market*, region, product_id, category, sub_category, product_name, sales, quantity, discount,
profit, shipping_cost*, order_priority*` — (*) only in the *Global* Superstore variant.

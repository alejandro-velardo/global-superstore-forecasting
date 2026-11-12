import pandas as pd
import pytest

from superstore.data.clean import audit, iqr_outlier_mask
from superstore.data.load import load_raw, parse_dates, to_snake


@pytest.mark.parametrize("raw,expected", [
    ("Sub-Category", "sub_category"), ("Row ID", "row_id"),
    ("Order Date", "order_date"), ("ShippingCost", "shipping_cost"),
])
def test_to_snake(raw, expected):
    assert to_snake(raw) == expected


def test_parse_dates_detects_dayfirst():
    df = pd.DataFrame({"order_date": ["03-04-2014", "25-12-2014"],
                       "ship_date": ["05-04-2014", "28-12-2014"]})
    out = parse_dates(df)
    assert out.loc[0, "order_date"] == pd.Timestamp("2014-04-03")


def test_parse_dates_detects_monthfirst():
    df = pd.DataFrame({"order_date": ["11/8/2016", "12/25/2016"],
                       "ship_date": ["11/11/2016", "12/28/2016"]})
    out = parse_dates(df)
    assert out.loc[0, "order_date"] == pd.Timestamp("2016-11-08")


def test_load_raw_latin1_csv(tmp_path):
    path = tmp_path / "superstore.csv"
    pd.DataFrame({"Order Date": ["01-02-2014"], "Ship Date": ["03-02-2014"],
                  "Customer Name": ["José Müller"], "Sales": [10.0]}
                 ).to_csv(path, index=False, encoding="latin-1")
    df = load_raw(path)
    assert {"order_date", "customer_name", "sales"} <= set(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])


def test_clean_orders_derived_fields(orders):
    assert {"profit_margin", "ship_days", "discount_band", "is_loss"} <= set(orders.columns)
    assert "row_id" not in orders.columns
    assert orders["order_date"].is_monotonic_increasing
    assert (orders["ship_days"] >= 0).all()


def test_audit_shape(raw):
    a = audit(raw)
    assert len(a) == raw.shape[1]
    assert {"pct_missing", "n_unique"} <= set(a.columns)


def test_iqr_mask():
    s = pd.Series([1, 2, 3, 4, 5, 100])
    assert iqr_outlier_mask(s).tolist() == [False] * 5 + [True]

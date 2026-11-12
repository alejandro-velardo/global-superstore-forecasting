import pytest

from superstore.data.clean import clean_orders
from superstore.data.load import optimise_dtypes, parse_dates, to_snake
from superstore.data.synthetic import make_synthetic_orders


@pytest.fixture(scope="session")
def raw():
    df = make_synthetic_orders(n=2000, seed=1)
    df.columns = [to_snake(c) for c in df.columns]
    return optimise_dtypes(parse_dates(df))


@pytest.fixture(scope="session")
def orders(raw):
    return clean_orders(raw)

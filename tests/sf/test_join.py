from datetime import datetime, timedelta
import polars as pl
from polars_ts.grouper import Grouper
import pytest


@pytest.fixture
def df() -> pl.LazyFrame:
    t = datetime(2024, 1, 1, 0)
    steps = list(range(0, 10, 3))
    nrows = len(steps)

    result = pl.LazyFrame(
        {
            "time": [t + timedelta(days=days + 1) for days in range(0, nrows * 2, 2)],
            "flt1": [2.18 + i for i in steps],
            "text": [f"stext {i}" for i in steps],
            "catsa": [f"A{i}" for i in steps],
            "catsb": [f"B{i % 3}" for i in steps],
            "flt2": [1.39 + i for i in steps],
        },
        schema={
            "time": pl.Datetime,
            "text": pl.String,
            "catsa": pl.Categorical,
            "catsb": pl.Categorical,
            "flt1": pl.Float64,
            "flt2": pl.Float32,
        },
    )

    return result


def test_join(df):
    g = Grouper()
    value_cols = g.values(df)

    self_join = df.sf.join(df).collect()
    expected_self_join = df.with_columns(
        [pl.col(c).alias(f"{c}_right") for c in value_cols]
    ).collect()
    assert self_join.select(sorted(self_join.columns)).equals(
        expected_self_join.select(sorted(expected_self_join.columns))
    )

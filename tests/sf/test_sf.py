from datetime import datetime, timedelta
import polars as pl
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


def test_unique(df):
    default_unique = (
        pl.concat([df.collect().lazy(), df.collect().lazy()]).sf.unique().collect()
    )
    expected_default_unique = df.collect()

    assert default_unique.equals(expected_default_unique)

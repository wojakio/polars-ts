from datetime import datetime, timedelta
import polars as pl
import pytest

from polars_ts.grouper import Grouper


@pytest.fixture
def df() -> pl.LazyFrame:
    t = datetime(2024, 1, 1, 0)
    nrows = 3
    result = pl.LazyFrame(
        {
            "time": [t]
            + [t + timedelta(days=days + 1) for days in range(0, nrows * 2, 2)],
            "flt1": [0] + [2.18 + i for i in range(nrows)],
            "text": ["X"] + [f"stext {i}" for i in range(nrows)],
            "catsa": ["CA"] + [f"A{i}" for i in range(nrows)],
            "catsb": ["CB"] + [f"B{i % 3}" for i in range(nrows)],
            "flt2": [0, 0] + [1.39 + i for i in range(nrows - 1)],
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


def test_basic(df):
    assert Grouper.by("catsa").apply(df) == ["catsa"]
    assert Grouper.by("catsa", "time").apply(df) == ["time", "catsa"]
    assert Grouper.by("catsa", "catsb").apply(df) == ["catsa", "catsb"]
    assert Grouper.by("time", "catsa", "catsb").apply(df) == ["time", "catsa", "catsb"]

    assert Grouper.by_time().apply(df) == ["time"]

    assert Grouper.by_time_and("catsa").apply(df) == ["time", "catsa"]
    assert Grouper.by_time_and("catsa", "catsb").apply(df) == ["time", "catsa", "catsb"]
    assert Grouper.by_time_and("time", "catsa").apply(df) == ["time", "catsa"]
    assert Grouper.by_time_and("time", "catsa", "catsb").apply(df) == [
        "time",
        "catsa",
        "catsb",
    ]

    assert Grouper.omitting("catsa").apply(df) == ["time", "catsb"]
    assert Grouper.omitting("catsa", "catsb").apply(df) == ["time"]
    assert Grouper.omitting("time", "catsa").apply(df) == ["catsb"]

    assert Grouper.omitting_time_and("catsa").apply(df) == ["catsb"]
    assert Grouper.omitting_time_and("time", "catsa").apply(df) == ["catsb"]

    assert Grouper.by_all().apply(df) == ["catsa", "catsb"]
    assert Grouper.by_time_and_all().apply(df) == ["time", "catsa", "catsb"]

    assert Grouper.by_common_excluding_time().apply(df, df) == ["catsa", "catsb"]
    assert Grouper.by_common_including_time().apply(df, df) == [
        "time",
        "catsa",
        "catsb",
    ]

    with pytest.raises(ValueError, match="yielded no columns"):
        Grouper.omitting_time_and("catsa", "catsb").apply(df) == []

    with pytest.raises(ValueError, match="yielded no columns"):
        Grouper.omitting("time", "catsa", "catsb").apply(df) == []


def test_default(df):
    with pytest.raises(
        ValueError, match="Bad Grouper invocation. Undefined specification"
    ):
        Grouper().apply(df)

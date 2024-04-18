from datetime import datetime, timedelta
import polars as pl
import pytest


@pytest.fixture
def df() -> pl.LazyFrame:
    t = datetime(2024, 1, 1, 0)
    steps = list(range(0, 10, 1))
    nrows = len(steps)

    result = pl.LazyFrame(
        {
            "time": [t + timedelta(days=days + 1) for days in range(0, nrows, 1)],
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


def test_basic(df):
    mean_alpha_1 = df.mathx.ewm_mean(half_life=1).collect()
    expected_mean_alpha_1 = pl.DataFrame(
        [
            pl.Series(
                "time",
                [
                    datetime(2024, 1, 2, 0, 0),
                    datetime(2024, 1, 3, 0, 0),
                    datetime(2024, 1, 4, 0, 0),
                    datetime(2024, 1, 5, 0, 0),
                    datetime(2024, 1, 6, 0, 0),
                    datetime(2024, 1, 7, 0, 0),
                    datetime(2024, 1, 8, 0, 0),
                    datetime(2024, 1, 9, 0, 0),
                    datetime(2024, 1, 10, 0, 0),
                    datetime(2024, 1, 11, 0, 0),
                ],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series(
                "text",
                [
                    "stext 0",
                    "stext 1",
                    "stext 2",
                    "stext 3",
                    "stext 4",
                    "stext 5",
                    "stext 6",
                    "stext 7",
                    "stext 8",
                    "stext 9",
                ],
                dtype=pl.String,
            ),
            pl.Series(
                "catsa",
                ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "catsb",
                ["B0", "B1", "B2", "B0", "B1", "B2", "B0", "B1", "B2", "B0"],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "flt1",
                [2.18, 3.18, 4.18, 5.18, 6.18, 7.18, 8.18, 9.18, 10.18, 11.18],
                dtype=pl.Float64,
            ),
            pl.Series(
                "flt2",
                [
                    1.3899999856948853,
                    2.390000104904175,
                    3.390000104904175,
                    4.389999866485596,
                    5.389999866485596,
                    6.389999866485596,
                    7.389999866485596,
                    8.390000343322754,
                    9.390000343322754,
                    10.390000343322754,
                ],
                dtype=pl.Float32,
            ),
        ]
    )

    assert mean_alpha_1.equals(expected_mean_alpha_1)

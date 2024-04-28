import datetime
import polars as pl
from polars.testing import assert_frame_equal
import polars_ts as ts  # noqa
import pytest


@pytest.fixture
def df() -> pl.LazyFrame:
    t = datetime.datetime(2024, 1, 1, 0).date()
    nrows = 10
    result = pl.LazyFrame(
        [
            pl.Series(
                "time",
                [
                    t + datetime.timedelta(days=days + 1)
                    for days in range(0, nrows * 2, 2)
                ],
            ),
            pl.Series(
                "item",
                ["A"] * (nrows // 2) + ["B"] * (nrows // 2),
                dtype=pl.Categorical,
            ),
            pl.Series("val1", list(range(0, nrows // 2)) * 2, dtype=pl.Float64) + 1,
            pl.Series("val2", list(range(0, nrows // 2)) * 2, dtype=pl.Float64) * 100.0
            + 100,
        ]
    )

    return result


@pytest.fixture
def params() -> pl.LazyFrame:
    result = pl.LazyFrame(
        [
            pl.Series("test_case", ["1", "2", "3", "4", "5"]),
            pl.Series("item", ["A", "A", "B", "B", "B"]),
            pl.Series("n", [1, 3, -1, -3, -50]),
        ]
    ).with_columns(pl.col(pl.String).cast(pl.Categorical))

    return result


def test_basic(df, params):
    result = df.mathx.shift(params).sort("test_case").collect()

    expected_result = (
        pl.DataFrame(
            [
                pl.Series(
                    "time",
                    [
                        datetime.date(2024, 1, 2),
                        datetime.date(2024, 1, 4),
                        datetime.date(2024, 1, 6),
                        datetime.date(2024, 1, 8),
                        datetime.date(2024, 1, 10),
                        datetime.date(2024, 1, 2),
                        datetime.date(2024, 1, 4),
                        datetime.date(2024, 1, 6),
                        datetime.date(2024, 1, 8),
                        datetime.date(2024, 1, 10),
                        datetime.date(2024, 1, 12),
                        datetime.date(2024, 1, 14),
                        datetime.date(2024, 1, 16),
                        datetime.date(2024, 1, 18),
                        datetime.date(2024, 1, 20),
                        datetime.date(2024, 1, 12),
                        datetime.date(2024, 1, 14),
                        datetime.date(2024, 1, 16),
                        datetime.date(2024, 1, 18),
                        datetime.date(2024, 1, 20),
                        datetime.date(2024, 1, 12),
                        datetime.date(2024, 1, 14),
                        datetime.date(2024, 1, 16),
                        datetime.date(2024, 1, 18),
                        datetime.date(2024, 1, 20),
                    ],
                    dtype=pl.Date,
                ),
                pl.Series(
                    "test_case",
                    [
                        "1",
                        "1",
                        "1",
                        "1",
                        "1",
                        "2",
                        "2",
                        "2",
                        "2",
                        "2",
                        "3",
                        "3",
                        "3",
                        "3",
                        "3",
                        "4",
                        "4",
                        "4",
                        "4",
                        "4",
                        "5",
                        "5",
                        "5",
                        "5",
                        "5",
                    ],
                    dtype=pl.Categorical(ordering="physical"),
                ),
                pl.Series(
                    "item",
                    [
                        "A",
                        "A",
                        "A",
                        "A",
                        "A",
                        "A",
                        "A",
                        "A",
                        "A",
                        "A",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                        "B",
                    ],
                    dtype=pl.Categorical(ordering="physical"),
                ),
                pl.Series(
                    "val2",
                    [
                        None,
                        100.0,
                        200.0,
                        300.0,
                        400.0,
                        None,
                        None,
                        None,
                        100.0,
                        200.0,
                        200.0,
                        300.0,
                        400.0,
                        500.0,
                        None,
                        400.0,
                        500.0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "val1",
                    [
                        None,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        None,
                        None,
                        None,
                        1.0,
                        2.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        None,
                        4.0,
                        5.0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                    dtype=pl.Float64,
                ),
            ]
        )
        .with_columns(pl.col("time").cast(pl.Date))
        .select(result.columns)
    )

    assert_frame_equal(result, expected_result)

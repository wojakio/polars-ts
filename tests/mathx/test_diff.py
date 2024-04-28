import datetime
import polars as pl
from polars.testing import assert_frame_equal
import pytest

import polars_ts as ts  # noqa


@pytest.fixture
def params() -> pl.LazyFrame:
    result = pl.LazyFrame(
        [
            pl.Series("method", ["arithmetic", "geometric", "fractional"]),
            pl.Series("n", [1, 1, 1]),
        ]
    ).with_columns(pl.col(pl.String).cast(pl.Categorical))

    return result


@pytest.fixture
def df():
    t = datetime.date(2024, 1, 1)
    nrows = 10

    factor = 1.1
    arithmetic_series = list(range(0, nrows))
    non_arithmetic_series = [1 * (factor**n) for n in range(0, nrows)]

    result = pl.LazyFrame(
        [
            pl.Series(
                "time",
                [t + datetime.timedelta(days=days + 1) for days in range(0, nrows)] * 3,
            ),
            pl.Series(
                "method",
                ["arithmetic"] * nrows + ["fractional"] * nrows + ["geometric"] * nrows,
                dtype=pl.Categorical,
            ),
            pl.Series(
                "value",
                arithmetic_series + non_arithmetic_series + non_arithmetic_series,
                pl.Float64,
            ),
        ]
    )

    return result


def test_basic(df, params):
    result_df = df.mathx.diff(params).collect()
    # print(result_df.to_init_repr())

    expected_df = pl.DataFrame(
        [
            pl.Series(
                "time",
                [
                    datetime.date(2024, 1, 2),
                    datetime.date(2024, 1, 3),
                    datetime.date(2024, 1, 4),
                    datetime.date(2024, 1, 5),
                    datetime.date(2024, 1, 6),
                    datetime.date(2024, 1, 7),
                    datetime.date(2024, 1, 8),
                    datetime.date(2024, 1, 9),
                    datetime.date(2024, 1, 10),
                    datetime.date(2024, 1, 11),
                    datetime.date(2024, 1, 2),
                    datetime.date(2024, 1, 3),
                    datetime.date(2024, 1, 4),
                    datetime.date(2024, 1, 5),
                    datetime.date(2024, 1, 6),
                    datetime.date(2024, 1, 7),
                    datetime.date(2024, 1, 8),
                    datetime.date(2024, 1, 9),
                    datetime.date(2024, 1, 10),
                    datetime.date(2024, 1, 11),
                    datetime.date(2024, 1, 2),
                    datetime.date(2024, 1, 3),
                    datetime.date(2024, 1, 4),
                    datetime.date(2024, 1, 5),
                    datetime.date(2024, 1, 6),
                    datetime.date(2024, 1, 7),
                    datetime.date(2024, 1, 8),
                    datetime.date(2024, 1, 9),
                    datetime.date(2024, 1, 10),
                    datetime.date(2024, 1, 11),
                ],
                dtype=pl.Date,
            ),
            pl.Series(
                "method",
                [
                    "arithmetic",
                    "arithmetic",
                    "arithmetic",
                    "arithmetic",
                    "arithmetic",
                    "arithmetic",
                    "arithmetic",
                    "arithmetic",
                    "arithmetic",
                    "arithmetic",
                    "fractional",
                    "fractional",
                    "fractional",
                    "fractional",
                    "fractional",
                    "fractional",
                    "fractional",
                    "fractional",
                    "fractional",
                    "fractional",
                    "geometric",
                    "geometric",
                    "geometric",
                    "geometric",
                    "geometric",
                    "geometric",
                    "geometric",
                    "geometric",
                    "geometric",
                    "geometric",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "value",
                [
                    None,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    None,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                ],
                dtype=pl.Float64,
            ),
        ]
    )

    assert_frame_equal(result_df, expected_df)

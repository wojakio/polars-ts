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


def test_basic_args_with_param(df) -> None:
    # with a param column
    result_df = df.mathx.diff(n=1, null_strategy="ignore").collect()
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
                    0.10000000000000009,
                    0.10000000000000007,
                    0.10000000000000017,
                    0.09999999999999996,
                    0.10000000000000007,
                    0.10000000000000014,
                    0.10000000000000016,
                    0.10000000000000013,
                    0.10000000000000009,
                    None,
                    1.1,
                    1.1,
                    1.1,
                    1.0999999999999999,
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


def test_basic_args_without_param(df) -> None:
    # with a param columns
    result_df = (
        df.filter(method="arithmetic")
        .drop("method")
        .mathx.diff(n=1, method="arithmetic", null_strategy="ignore")
        .collect()
    )

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
                ],
                dtype=pl.Date,
            ),
            pl.Series(
                "value",
                [None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=pl.Float64,
            ),
        ]
    )

    assert_frame_equal(result_df, expected_df)


def test_basic_params(df, params):
    result_df = df.mathx.diff(params=params).collect()
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
                    0.10000000000000009,
                    0.10000000000000007,
                    0.10000000000000017,
                    0.09999999999999996,
                    0.10000000000000007,
                    0.10000000000000014,
                    0.10000000000000016,
                    0.10000000000000013,
                    0.10000000000000009,
                    None,
                    1.1,
                    1.1,
                    1.1,
                    1.0999999999999999,
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


def test_multi_param_no_categories():
    df = pl.LazyFrame(
        [
            pl.Series("X", ["u"] * 4 + ["v"] * 4, dtype=pl.Categorical),
            pl.Series("value", list(range(0, 4)) + list(range(0, 4)), dtype=pl.Int64),
        ]
    )

    params = pl.LazyFrame(
        [
            pl.Series("a", ["diff_1", "diff_2"], dtype=pl.Categorical),
            pl.Series("n", [1, 2], dtype=pl.Int64),
            pl.Series("null_strategy", ["drop_if_any", "ignore"], dtype=pl.Categorical),
        ]
    )

    with pytest.raises(ValueError, match="multiple params without any categories"):
        (df.mathx.diff(params=params.drop("a")).collect())

    result_df = df.mathx.diff(params=params).collect()

    # print(result_df.to_init_repr())
    expected_df = pl.DataFrame(
        [
            pl.Series(
                "X",
                ["u", "u", "u", "u", "u", "u", "u", "v", "v", "v", "v", "v", "v", "v"],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "value",
                [1, 1, 1, None, None, 2, 2, 1, 1, 1, None, None, 2, 2],
                dtype=pl.Int64,
            ),
            pl.Series(
                "a",
                [
                    "diff_1",
                    "diff_1",
                    "diff_1",
                    "diff_2",
                    "diff_2",
                    "diff_2",
                    "diff_2",
                    "diff_1",
                    "diff_1",
                    "diff_1",
                    "diff_2",
                    "diff_2",
                    "diff_2",
                    "diff_2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
        ]
    )
    assert_frame_equal(result_df, expected_df)

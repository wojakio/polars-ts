from datetime import datetime, timedelta
import polars as pl
from polars.testing import assert_frame_equal
import pytest


@pytest.fixture
def arithmetic_df():
    t = datetime(2024, 1, 1).date()
    steps = list(range(0, 100, 5))
    nrows = len(steps)

    result = pl.LazyFrame(
        [
            pl.Series(
                "t", [t + timedelta(days=days + 1) for days in range(0, nrows * 2, 2)]
            ),
            pl.Series("n", list(range(0, nrows))),
        ]
    )

    return result


@pytest.fixture
def fractional_df():
    t = datetime(2024, 1, 1).date()
    steps = list(range(0, 100, 5))
    nrows = len(steps)
    factor = 1.1

    result = pl.LazyFrame(
        [
            pl.Series(
                "t", [t + timedelta(days=days + 1) for days in range(0, nrows * 2, 2)]
            ),
            pl.Series("n", [1 * (factor**n) for n in range(0, nrows)]),
        ]
    )

    return result


@pytest.fixture
def geometric_df():
    t = datetime(2024, 1, 1).date()
    steps = list(range(0, 100, 5))
    nrows = len(steps)
    factor = 1.1

    result = pl.LazyFrame(
        [
            pl.Series(
                "t", [t + timedelta(days=days + 1) for days in range(0, nrows * 2, 2)]
            ),
            pl.Series("n", [1 * (factor**n) for n in range(0, nrows)]),
        ]
    )

    return result


def test_arithmetic_method(arithmetic_df):
    result_df = arithmetic_df.mathx.diff(1, method="arithmetic").collect()

    expected_df = pl.DataFrame(
        [
            arithmetic_df.collect()["t"].slice(1),
            pl.Series(
                "n",
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                dtype=pl.Int64,
            ),
        ]
    )

    assert_frame_equal(result_df, expected_df)

    result_df = arithmetic_df.mathx.diff(2, method="arithmetic").collect()

    expected_df = pl.DataFrame(
        [
            arithmetic_df.collect()["t"].slice(2),
            pl.Series(
                "n",
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                dtype=pl.Int64,
            ),
        ]
    )

    assert_frame_equal(result_df, expected_df)


def test_geometric_method(geometric_df):
    result_df = geometric_df.mathx.diff(1, method="geometric").collect()

    expected_df = pl.DataFrame(
        [
            geometric_df.collect()["t"].slice(1),
            pl.Series(
                "n",
                [
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
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

    result_df = geometric_df.mathx.diff(2, method="geometric").collect()

    expected_df = pl.DataFrame(
        [
            geometric_df.collect()["t"].slice(2),
            pl.Series(
                "n",
                [
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                    1.21,
                ],
                dtype=pl.Float64,
            ),
        ]
    )

    assert_frame_equal(result_df, expected_df)


def test_fractional_method(fractional_df):
    result_df = fractional_df.mathx.diff(1, method="fractional").collect()

    expected_df = pl.DataFrame(
        [
            fractional_df.collect()["t"].slice(1),
            pl.Series(
                "n",
                [
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                ],
                dtype=pl.Float64,
            ),
        ]
    )

    assert_frame_equal(result_df, expected_df)

    result_df = fractional_df.mathx.diff(2, method="fractional").collect()

    expected_df = pl.DataFrame(
        [
            fractional_df.collect()["t"].slice(2),
            pl.Series(
                "n",
                [
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                ],
                dtype=pl.Float64,
            ),
        ]
    )

    assert_frame_equal(result_df, expected_df)

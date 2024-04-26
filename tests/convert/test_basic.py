import polars as pl
from polars.testing import assert_frame_equal
import polars_ts as ts  # noqa
import pytest

from ..config import get_config_filename


@pytest.fixture
def physical_conversions() -> pl.LazyFrame:
    filename = get_config_filename("physical_conversions.csv")
    df = pl.LazyFrame().io.read_csv(filename)  # type: ignore[attr-defined]
    return df


@pytest.fixture
def currency_conversions() -> pl.LazyFrame:
    filename = get_config_filename("currency_conversions.csv")
    df = pl.LazyFrame().io.read_csv(filename)  # type: ignore[attr-defined]
    return df


def test_single_unit(physical_conversions):
    physical_conversions = physical_conversions.convert.construct_closure()

    df = (
        pl.from_records(
            data=[
                ["tonne", 1.0],
                ["stone", 1.0],
                ["mm", 1000.0],
                ["m", 1.0],
            ],
            orient="row",
            schema=["value_unit", "value"],
        )
        .with_columns(pl.col("value_unit").cast(pl.Categorical))
        .lazy()
    )

    # -----------------------------------------------------------------------
    # test unknown unit
    result_df = df.convert.convert(
        pl.lit("unknown_unit"), physical_conversions
    ).collect()
    assert result_df.equals(df.collect())

    # -----------------------------------------------------------------------
    # test actual conversion via literal argument

    result_df = (
        df.convert.convert(pl.lit("nm"), physical_conversions)
        .convert.convert(pl.lit("stone"), physical_conversions)
        .collect()
    )

    expected_df = pl.DataFrame(
        [
            pl.Series(
                "value_unit",
                ["stone", "stone", "nm", "nm"],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "value", [157.473, 1.0, 1000000000.0, 1000000000.0], dtype=pl.Float64
            ),
        ]
    )

    assert_frame_equal(result_df, expected_df)

    # -----------------------------------------------------------------------
    # test actual conversion via column argument
    df_with_target_col = df.with_columns(
        pl.Series("target_unit", ["stone", "gram", "miles", "nm"], dtype=pl.Categorical)
    )

    result_df = df_with_target_col.convert.convert(
        "target_unit", physical_conversions
    ).collect()

    expected_df = pl.DataFrame(
        [
            pl.Series(
                "value_unit",
                ["stone", "gram", "miles", "nm"],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "value",
                [157.473, 6350.294971201412, 0.000621371, 1000000000.0],
                dtype=pl.Float64,
            ),
            pl.Series(
                "target_unit",
                ["stone", "gram", "miles", "nm"],
                dtype=pl.Categorical(ordering="physical"),
            ),
        ]
    )

    assert_frame_equal(result_df, expected_df)

    # -----------------------------------------------------------------------
    # test actual conversion via column argument, and alternate column names
    df_renamed = df_with_target_col.rename(
        {
            "value": "given",
            "value_unit": "given_unit",
            "target_unit": "desired_unit",
        }
    )

    result_df_1 = df_renamed.convert.convert(
        "desired_unit", physical_conversions, "given", "given_unit"
    ).collect()

    result_df_2 = df_renamed.convert.convert(
        "desired_unit",
        physical_conversions,
        "given",
    ).collect()

    expected_renamed_df = expected_df.rename(
        {
            "value": "given",
            "value_unit": "given_unit",
            "target_unit": "desired_unit",
        }
    )

    assert_frame_equal(result_df_1, expected_renamed_df)
    assert_frame_equal(result_df_2, expected_renamed_df)


def test_dual_unit(physical_conversions):
    physical_conversions = physical_conversions.convert.construct_closure()

    df = (
        pl.from_records(
            data=[
                ["miles/hr", 1.0],
                ["km/sec", 1.0],
            ],
            orient="row",
            schema=["value_unit", "value"],
        )
        .with_columns(pl.col("value_unit").cast(pl.Categorical))
        .lazy()
    )

    # -----------------------------------------------------------------------
    # test unknown unit
    result_df = df.convert.convert(
        pl.lit("unknown_numerator/unknown_denominator"), physical_conversions
    ).collect()
    assert result_df.equals(df.collect())

    # -----------------------------------------------------------------------
    # test actual conversion via literal argument

    result_df = df.convert.convert(
        pl.lit("km/hr"), physical_conversions, is_multi_dim=True
    ).collect()

    expected_df = pl.DataFrame(
        [
            pl.Series("value_unit", ["km/hr", "km/hr"], dtype=pl.String),
            pl.Series("value", [1.6093444978925633, 3600.0], dtype=pl.Float64),
        ]
    )

    assert_frame_equal(result_df, expected_df)

    # -----------------------------------------------------------------------
    # test actual conversion via column argument
    df_with_target_col = df.with_columns(
        pl.Series("target_unit", ["cm/sec", "miles/day"], dtype=pl.Categorical)
    )

    result_df = df_with_target_col.convert.convert(
        "target_unit", physical_conversions, is_multi_dim=True
    ).collect()

    expected_df = pl.DataFrame(
        [
            pl.Series("value_unit", ["cm/sec", "miles/day"], dtype=pl.String),
            pl.Series("value", [44.70401383034898, 53686.4544], dtype=pl.Float64),
            pl.Series(
                "target_unit",
                ["cm/sec", "miles/day"],
                dtype=pl.Categorical(ordering="physical"),
            ),
        ]
    )

    assert_frame_equal(result_df, expected_df)

    # -----------------------------------------------------------------------
    # test actual conversion via column argument, and alternate column names
    df_renamed = df_with_target_col.rename(
        {
            "value": "given",
            "value_unit": "given_unit",
            "target_unit": "desired_unit",
        }
    )

    result_df_1 = df_renamed.convert.convert(
        "desired_unit", physical_conversions, "given", "given_unit", is_multi_dim=True
    ).collect()

    result_df_2 = df_renamed.convert.convert(
        "desired_unit", physical_conversions, "given", is_multi_dim=True
    ).collect()

    expected_renamed_df = expected_df.rename(
        {
            "value": "given",
            "value_unit": "given_unit",
            "target_unit": "desired_unit",
        }
    )

    assert_frame_equal(result_df_1, expected_renamed_df)
    assert_frame_equal(result_df_2, expected_renamed_df)

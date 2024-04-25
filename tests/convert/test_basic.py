import polars as pl
import polars_ts as ts # noqa
import pytest

from ..config import get_config_filename


@pytest.fixture
def physics_conversions() -> pl.LazyFrame:
    filename = get_config_filename("physics_conversions.csv")
    df = pl.LazyFrame().io.read_csv(filename)
    return df

@pytest.fixture
def currency_conversions() -> pl.LazyFrame:
    filename = get_config_filename("currency_conversions.csv")
    df = pl.LazyFrame().io.read_csv(filename)
    return df


def test_single_unit(physics_conversions):
    physics_conversions = physics_conversions.convert.construct_closure()
    
    df = (
        pl.from_records(
            data=[
                ["tonne", 1.0],
                ["stone", 1.0],
                ["mm", 1000.0],
                ["m", 1.0],
            ],        
            orient='row',
            schema=["value_unit", "value"],
        )
        .with_columns(pl.col("value_unit").cast(pl.Categorical))
        .lazy()
    )

    with pytest.raises(ValueError, match="Unknown target_unit"):
        _err_df = df.convert.convert("bad_unit", physics_conversions).collect()

    result_df = (
        df
        .convert.convert("m", physics_conversions)
        .convert.convert("stone", physics_conversions)
        .collect()
    )

    expected_df = pl.DataFrame(
        [
            pl.Series("value_unit", ['stone', 'stone', 'm', 'm'], dtype=pl.Categorical(ordering='physical')),
            pl.Series("value", [157.473, 1.0, 1.0, 1.0], dtype=pl.Float64),
        ]
    )


    assert result_df.equals(expected_df)


    result_df = (
        df
        .convert.convert("nm", physics_conversions)
        .convert.convert("stone", physics_conversions)
        .collect()
    )

    expected_df = pl.DataFrame(
        [
            pl.Series("value_unit", ['stone', 'stone', 'nm', 'nm'], dtype=pl.Categorical(ordering='physical')),
            pl.Series("value", [157.473, 1.0, 1000000000.0, 1000000000.0], dtype=pl.Float64),
        ]
    )

    assert result_df.equals(expected_df)


def test_dual_unit_numerator(currency_conversions):

    df = (
        pl.from_records(
            data=[
                ["miles/hr", 1.0],
                ["km/sec", 1.0],
                ["mm", 1000.0],
                ["m", 1.0],
            ],        
            orient='row',
            schema=["value_unit", "value"],
        )
        .with_columns(pl.col("value_unit").cast(pl.Categorical))
        .lazy()
    )

    result_df = (
        df
        .convert.convert("km/hr", physics_conversions)
        # .collect()
    )

    assert df is not None

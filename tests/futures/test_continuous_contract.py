from datetime import datetime, UTC

import polars as pl
import polars_ts  # noqa

from ..config import get_config_filename
from polars_ts.futures_helper.roll_calendar import guess_security_meta
from polars_ts.futures_helper.generic_symbols import generic_symbol_universe


def test_continuous_contract():
    market_start_date = datetime(1990, 1, 1, 23, 0, tzinfo=UTC)
    market_end_date = datetime(2000, 1, 1, 23, 0, tzinfo=UTC)

    roll_config_filename = get_config_filename("roll_config.csv")
    roll_config = pl.LazyFrame().io.read_csv(roll_config_filename)

    generic_symbols = generic_symbol_universe(
        roll_config, market_start_date, market_end_date
    )
    security_meta = guess_security_meta(generic_symbols, roll_config)

    security_dates = (
        security_meta.select(
            "asset",
            "instrument_id",
            "fut_first_trade_dt",
            expiry_date=pl.min_horizontal("last_tradeable_dt", "fut_notice_first"),
        )
        .collect()
        .lazy()
    )

    roll_calendar = pl.LazyFrame().future.create_roll_calendar(
        roll_config, security_dates
    )

    instrument_prices = security_dates.dummymkt.fetch_instrument_prices(
        market_start_date.date(),
        market_end_date.date(),
        remove_weekends=True,
    )

    empty_adjusted_prices = (
        pl.LazyFrame()
        .future.prepare_unadjusted_for_stitching(
            roll_calendar, instrument_prices.clear()
        )
        .future.stitch_panama_backwards()
    )

    adjusted_prices = (
        pl.LazyFrame()
        .future.prepare_unadjusted_for_stitching(roll_calendar, instrument_prices)
        .future.stitch_panama_backwards()
    )

    adjusted_prices_schema = list(adjusted_prices.schema.items())

    expected_schema = [
        ("time", pl.Date),
        ("stitching", pl.Categorical(ordering="physical")),
        ("asset", pl.Categorical(ordering="physical")),
        (
            "instrument_id",
            pl.Struct({"tenor": pl.Categorical(ordering="physical"), "year": pl.Int32}),
        ),
        ("value", pl.Float64),
    ]

    assert adjusted_prices_schema == expected_schema

    result = adjusted_prices.collect()
    # instrument_prices.collect().write_parquet(get_data_filename("futures", "instrument_prices.parquet"))
    # adjusted_prices.collect().write_parquet(get_data_filename("futures", "adjusted_prices.parquet"))
    # expected_instrument_prices = pl.read_parquet(get_data_filename("futures", "instrument_prices.parquet"))
    # expected_adjusted_prices = pl.read_parquet(get_data_filename("futures", "adjusted_prices.parquet"))
    # assert result.equals(expected_result)

    assert result.shape == (15660, 5)
    assert instrument_prices.collect()["value"].sum() == 926966.0052603701
    assert adjusted_prices.collect()["value"].sum() == -121883.7423864497

    empty_result = empty_adjusted_prices.collect()
    assert list(empty_result.schema.items()) == expected_schema
    assert empty_result.is_empty()

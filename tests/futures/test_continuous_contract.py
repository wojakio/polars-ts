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
    roll_config = pl.LazyFrame().loader.csv(roll_config_filename)

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
        "fut_first_trade_dt",
        pl.min_horizontal("expiry_date", pl.lit(market_end_date.date())),
        remove_weekends=False,
    )

    adjusted_prices = (
        pl.LazyFrame()
        .future.prepare_unadjusted_for_stitching(roll_calendar, instrument_prices)
        .future.stitch_panama_backwards()
    )

    adjusted_prices_schema = list(adjusted_prices.schema.items())

    expected_schema = [
        ("time", pl.Date),
        ("asset", pl.Categorical(ordering="physical")),
        (
            "instrument_id",
            pl.Struct({"tenor": pl.Categorical(ordering="physical"), "year": pl.Int32}),
        ),
        ("unadjusted", pl.Float64),
        ("panama_backwards", pl.Float64),
    ]

    assert adjusted_prices_schema == expected_schema

    assert adjusted_prices.shape == (10030, 5)

    # replace that with a stable hash
    assert adjusted_prices.hash_rows().sum() == 4148443207706655575

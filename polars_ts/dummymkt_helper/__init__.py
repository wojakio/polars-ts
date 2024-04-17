import polars as pl

from ..types import FrameType
from ..dummy_helper import impl_random_normal
from ..grouper import Grouper


def impl_fetch_instrument_prices(
    df: FrameType, start_dt: pl.Expr, end_dt: pl.Expr, remove_weekends: bool
) -> FrameType:
    hols = [6, 7] if remove_weekends else []

    result = (
        df.select(
            time=pl.date_ranges(start_dt, end_dt),
            asset=pl.col("asset"),
            instrument_id=pl.col("instrument_id"),
        )
        .explode(pl.col("time"))
        .pipe(
            lambda df: impl_random_normal(
                df, partition=Grouper(), out="value", mu=0.0, sigma=1.0
            )
        )
        .filter(pl.col("time").dt.weekday().is_in(hols).not_())
        .with_columns(pl.col("value").cum_sum().over("asset", "instrument_id"))
    )

    return result


def impl_fetch_roll_calendar_prices(
    roll_calendar: FrameType, instrument_prices: FrameType
) -> FrameType:
    prices = instrument_prices.rename({"time": "roll_date"})

    df = (
        roll_calendar.join(
            prices.rename({"instrument_id": "near_contract", "value": "near_price"}),
            on=["roll_date", "asset", "near_contract"],
            how="left",
        )
        .join(
            prices.rename({"instrument_id": "far_contract", "value": "far_price"}),
            on=["roll_date", "asset", "far_contract"],
            how="left",
        )
        .join(
            prices.rename({"instrument_id": "carry_contract", "value": "carry_price"}),
            on=["roll_date", "asset", "carry_contract"],
            how="left",
        )
    )

    return df

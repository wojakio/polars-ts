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
                df, partition=Grouper.by_all(), out="value", mu=0.0, sigma=1.0
            )
        )
        .filter(pl.col("time").dt.weekday().is_in(hols).not_())
        .with_columns(pl.col("value").cum_sum().over("asset", "instrument_id"))
    )

    return result


def _roll_back_missing_stitch_points(
    df: FrameType, prices: FrameType, lookback_interval: str
) -> FrameType:
    has_missing_price = pl.any_horizontal(
        pl.col("near_price").is_null(),
        pl.col("far_price").is_null(),
    )

    missing_prices = df.filter(has_missing_price)

    candidate_roll_data = (
        missing_prices.select(
            "asset",
            "near_contract",
            "far_contract",
            "carry_contract",
            start_date=pl.col("roll_date").dt.offset_by(f"-{lookback_interval}"),
            end_date="roll_date",
        )
        .with_columns(roll_date=pl.date_ranges("start_date", "end_date", "1d"))
        .drop("start_date", "end_date")
        .explode("roll_date")
        .join(
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
        .sort("asset", "roll_date")
    )

    updated_rolls = (
        candidate_roll_data.filter(has_missing_price.not_())
        .group_by(
            "asset",
            "near_contract",
            "far_contract",
            "carry_contract",
            maintain_order=True,
        )
        .last()
    )

    result = df.update(
        updated_rolls,
        on=["asset", "near_contract", "far_contract", "carry_contract"],
        how="left",
        include_nulls=True,
    )

    return result


def impl_fetch_roll_calendar_prices(
    roll_calendar: FrameType,
    instrument_prices: FrameType,
    stitch_lookback_interval: str,
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

    df = _roll_back_missing_stitch_points(df, prices, stitch_lookback_interval)

    return df

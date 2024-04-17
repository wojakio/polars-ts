import polars as pl

from ..types import FrameType
from .roll_calendar import create_roll_calendar_helper


def impl_create_roll_calendar(
    roll_config: FrameType,
    security_expiries: FrameType,
    include_debug: bool = False,
) -> FrameType:
    result = create_roll_calendar_helper(roll_config, security_expiries, include_debug)

    return result


def impl_prepare_unadjusted_for_stitching(
    df: FrameType,
    roll_calendar: FrameType,
    price_universe: FrameType,
) -> FrameType:
    far_contract_prices = (
        df.select()
        .dummymkt.fetch_roll_calendar_prices(roll_calendar, price_universe)  # type: ignore[attr-defined]
        .select(
            "asset",
            time="roll_date",
            instrument_id="near_contract",
            next_instrument_id="far_contract",
            next_value="far_price",
        )
    )

    result = (
        roll_calendar.select(
            "asset",
            instrument_id="near_contract",
            instrument_start_date=pl.col("roll_date")
            .shift(1, fill_value=pl.min("roll_date").dt.offset_by("-5y"))
            .over("asset"),
            instrument_end_date="roll_date",
        )
        .join(price_universe, on=["asset", "instrument_id"], how="left")
        .join(far_contract_prices, on=["time", "asset", "instrument_id"], how="left")
        .filter(
            pl.col("time")
            .is_between("instrument_start_date", "instrument_end_date", "right")
            .over("instrument_id")
        )
        .select(pl.exclude("instrument_start_date", "instrument_end_date"))
    )

    return result


def impl_stitch_panama_backwards(df: FrameType) -> FrameType:
    result = (
        df.with_columns(
            roll_adj=(
                pl.when(pl.col("next_value").is_not_null())
                .then(pl.col("next_value") - pl.col("value"))
                .otherwise(0.0)
            )
        )
        .with_columns(
            cum_adj=(pl.col("roll_adj").reverse().cum_sum().reverse().over("asset"))
        )
        .with_columns(adj=pl.col("value") + pl.col("cum_adj"))
        # cleanup
        .select(
            "time",
            "asset",
            "instrument_id",
            unadjusted="value",
            panama_backwards="adj",
        )
    )

    return result

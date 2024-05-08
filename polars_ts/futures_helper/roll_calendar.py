import polars as pl

from .util import month_to_imm_dict, make_generic_contract
from ..types import FrameType


def guess_security_meta(
    generic_contracts: FrameType, roll_config: FrameType
) -> FrameType:
    df = (
        roll_config.join(generic_contracts, on="asset")
        .filter(
            pl.col("priced_roll_cycle").str.contains(
                pl.col("instrument_id").struct.field("tenor").cast(pl.String)
            )
        )
        .with_columns(
            expiry_date=(
                pl.col("time")
                .dt.month_start()
                .dt.offset_by(pl.col("approximate_expiry_offset").cast(pl.String))
                .dt.date()
            )
        )
        .select(
            "asset",
            "instrument_id",
            fut_first_trade_dt=_guess_fut_first_trade_dt(
                pl.col("expiry_date"), pl.col("approximate_contract_lifespan")
            ),
            last_tradeable_dt="expiry_date",
            fut_notice_first="expiry_date",
        )
        .unique(maintain_order=True)
    )

    return df


def _guess_fut_first_trade_dt(expiry_date: pl.Expr, trade_lifespan: pl.Expr) -> pl.Expr:
    first_trade_dt = pl.min_horizontal(
        (
            expiry_date.dt.month_start()
            .dt.offset_by(pl.col("approximate_expiry_offset").cast(pl.String))
            .dt.offset_by(pl.col("roll_offset").cast(pl.String))
        ),
        (expiry_date.dt.offset_by(pl.concat_str(pl.lit("-"), trade_lifespan))),
    )

    return first_trade_dt


def asset_carry_contracts(roll_config: FrameType) -> FrameType:
    def _find_carry_month(hc: dict):
        allowed_contracts = hc["priced_roll_cycle"]
        current_contract = hc["hold_roll_cycle"]
        carry_offset = hc["carry_contract_offset"]

        idx = allowed_contracts.find(current_contract)
        carry_idx = (idx + carry_offset) % len(allowed_contracts)
        carry_imm_month = allowed_contracts[carry_idx]

        return carry_imm_month

    return (
        roll_config.select(
            "asset",
            "carry_contract_offset",
            pl.col("hold_roll_cycle").cast(pl.String).str.extract_all("[F-Z]"),
            pl.col("priced_roll_cycle").cast(pl.String),
        )
        .explode("hold_roll_cycle")
        .with_columns(
            hold_month=pl.col("hold_roll_cycle")
            .replace(month_to_imm_dict(invert=True))
            .cast(pl.Int8),
            carry_month=(
                pl.struct(
                    "hold_roll_cycle", "carry_contract_offset", "priced_roll_cycle"
                )
                .map_elements(_find_carry_month, return_dtype=pl.String)
                .replace(month_to_imm_dict(invert=True))
                .cast(pl.Int8)
            ),
        )
        .with_columns(carry_month_offset=pl.col("carry_month") - pl.col("hold_month"))
        .with_columns(
            carry_month_offset=(
                pl.concat_str(
                    [
                        pl.when(
                            (pl.col("carry_contract_offset") < 0)
                            & (pl.col("carry_month_offset") > 0)
                        )
                        .then(pl.col("carry_month_offset") - 12)
                        .when(
                            (pl.col("carry_contract_offset") > 0)
                            & (pl.col("carry_month_offset") < 0)
                        )
                        .then(pl.col("carry_month_offset") + 12)
                        .otherwise(pl.col("carry_month_offset")),
                        pl.lit("mo"),
                    ]
                )
            )
        )
        .select("asset", "hold_month", "carry_month_offset")
    )


def asset_near_far_contracts(roll_config: FrameType) -> FrameType:
    return (
        roll_config.select(
            "asset",
            hold_near=pl.col("hold_roll_cycle")
            .cast(pl.String)
            .str.extract_all("[F-Z]"),
            hold_far=pl.concat_str(
                pl.col("hold_roll_cycle").cast(pl.String).str.slice(1),
                pl.col("hold_roll_cycle").cast(pl.String).str.slice(0, 1),
            ).str.extract_all("[F-Z]"),
        )
        .explode("hold_near", "hold_far")
        .with_columns(
            hold_near_month=pl.col("hold_near")
            .replace(month_to_imm_dict(invert=True))
            .cast(pl.Int8),
            hold_far_month=pl.col("hold_far")
            .replace(month_to_imm_dict(invert=True))
            .cast(pl.Int8),
        )
        .with_columns(
            hold_far_month_offset=pl.col("hold_far_month") - pl.col("hold_near_month"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.when(pl.col("hold_far_month_offset") <= 0)
                    .then(pl.col("hold_far_month_offset") + 12)
                    .otherwise(pl.col("hold_far_month_offset")),
                    pl.lit("mo"),
                ]
            )
        )
    )


def asset_contracts(roll_config: FrameType) -> FrameType:
    return (
        asset_near_far_contracts(roll_config)
        .join(
            asset_carry_contracts(roll_config),
            left_on=[pl.col("asset"), pl.col("hold_near_month")],
            right_on=[pl.col("asset"), pl.col("hold_month")],
            how="left",
        )
        .select(
            "asset", "hold_near_month", "hold_far_month_offset", "carry_month_offset"
        )
    )


def create_roll_calendar_helper(
    roll_config: FrameType, security_dates: FrameType, include_debug: bool
) -> FrameType:
    result_schema = [
        "roll_date",
        "asset",
        "near_contract",
        "far_contract",
        "carry_contract",
    ]

    if include_debug:
        result_schema.extend(
            [
                "has_instruments",
            ]
        )

    imm2mo_dict = month_to_imm_dict(invert=True)

    df = (
        roll_config.join(security_dates, on="asset", how="left")
        .select(
            "asset",
            "instrument_id",
            has_instruments=pl.col("expiry_date").is_not_null(),
            roll_date=(
                pl.col("expiry_date").dt.offset_by(
                    pl.col("roll_offset").cast(pl.String)
                )
            ),
        )
        .with_columns(
            hold_near_month=(
                pl.col("instrument_id")
                .struct.field("tenor")
                .cast(pl.String)
                .replace(imm2mo_dict)
                .cast(pl.Int8)
            )
        )
        .join(
            asset_contracts(roll_config), on=["asset", "hold_near_month"], how="inner"
        )
        .sort("asset", "roll_date")
        .with_columns(
            pl.col("hold_near_month", "hold_far_month_offset", "carry_month_offset")
            .backward_fill()
            .over("asset")
        )
        .with_columns(
            near_contract_dt=pl.date(
                pl.col("instrument_id").struct.field("year"),
                pl.col("hold_near_month"),
                1,
            )
        )
        .with_columns(
            far_contract_dt=pl.col("near_contract_dt").dt.offset_by(
                pl.col("hold_far_month_offset").cast(pl.String)
            )
        )
        .with_columns(
            carry_contract_dt=pl.col("near_contract_dt").dt.offset_by(
                pl.col("carry_month_offset").cast(pl.String)
            )
        )
        .with_columns(
            near_contract=make_generic_contract("near_contract_dt"),
            far_contract=make_generic_contract("far_contract_dt"),
            carry_contract=make_generic_contract("carry_contract_dt"),
        )
        .select(result_schema)
        # TODO: change this to drop nulls at end of table
        .drop_nulls()
    )

    return df

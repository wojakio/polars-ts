from datetime import datetime
import polars as pl

from .util import month_to_imm_dict

def guess_security_meta(
    bbg_symbols: pl.LazyFrame,
    roll_config: pl.LazyFrame
) -> pl.LazyFrame:

    df = (
        roll_config
        .join(bbg_symbols, on="asset")
        .filter(pl.col("priced_roll_cycle").str.contains(pl.col("imm").cast(pl.String)))
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
            "ticker",
            fut_first_trade_dt=_guess_fut_first_trade_dt(pl.col("expiry_date")),
            last_tradeable_dt="expiry_date",
            fut_notice_first="expiry_date",
        )
        
        .unique(maintain_order=True)
    )

    return df


def _guess_fut_first_trade_dt(
        expiry_date: pl.Expr,
        min_trade_days: str = "-2y",
        buffer_days: str = "-30d"
) -> pl.Expr:
    
    first_trade_dt = pl.min_horizontal(
        (
            expiry_date
            .dt.month_start()
            .dt.offset_by(pl.col("approximate_expiry_offset").cast(pl.String))
            .dt.offset_by(pl.col("roll_offset").cast(pl.String))
            .dt.offset_by(buffer_days)
        ),
        (
            expiry_date.dt.offset_by(min_trade_days)
        )
    )

    return first_trade_dt

    

def asset_carry_contracts(roll_config: pl.LazyFrame) -> pl.LazyFrame:
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


def asset_near_far_contracts(roll_config: pl.LazyFrame) -> pl.LazyFrame:
    return (
        roll_config.select(
            "asset",
            hold_near=pl.col("hold_roll_cycle").cast(pl.String).str.extract_all("[F-Z]"),
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


def asset_contracts(roll_config: pl.LazyFrame) -> pl.LazyFrame:
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
    roll_config: pl.LazyFrame,
    security_expiries: pl.LazyFrame,
    include_debug: bool
) -> pl.LazyFrame:
    
    result_schema = [
        "roll_date",
        "asset",
        "near_contract",
        "far_contract",
        "carry_contract",
    ]

    if include_debug:
        result_schema.extend([
            "has_tickers",
        ])

    
    df = (
        roll_config.select("asset", "roll_offset")
        .join(security_expiries, on="asset", how="left")
        .with_columns(
            has_tickers=pl.col("expiry_date").is_not_null(),
            roll_date=(
                pl.col("expiry_date").dt.offset_by(pl.col("roll_offset").cast(pl.String)).dt.date()
            )
        )
        .with_columns(hold_near_month=pl.col("expiry_date").dt.month())
        .join(asset_contracts(roll_config), on=["asset", "hold_near_month"], how="left")
        .with_columns(
            pl.col("hold_near_month", "hold_far_month_offset", "carry_month_offset")
            .backward_fill()
            .over("asset")
        )
        .with_columns(
            near_contract=pl.date(
                pl.col("expiry_date").dt.year(), pl.col("hold_near_month"), 1
            )
        )
        .with_columns(
            far_contract=pl.col("near_contract").dt.offset_by(
                pl.col("hold_far_month_offset").cast(pl.String)
            )
        )
        .with_columns(
            carry_contract=pl.col("near_contract").dt.offset_by(
                pl.col("carry_month_offset").cast(pl.String)
            )
        )
        .select(result_schema)
        .sort("asset", "roll_date")

        # TODO: change this to drop nulls at end of table
        .drop_nulls()
    )

    return df

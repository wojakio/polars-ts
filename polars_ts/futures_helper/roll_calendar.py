import polars as pl

from .util import month_to_imm


def load_roll_config(filename: str) -> pl.LazyFrame:
    return pl.scan_csv(filename)


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
            pl.col("hold_roll_cycle").str.extract_all("[F-Z]"),
            pl.col("priced_roll_cycle"),
        )
        .explode("hold_roll_cycle")
        .with_columns(
            hold_month=pl.col("hold_roll_cycle")
            .replace(month_to_imm(as_dict=True, invert=True))
            .cast(pl.Int8),
            carry_month=(
                pl.struct(
                    "hold_roll_cycle", "carry_contract_offset", "priced_roll_cycle"
                )
                .map_elements(_find_carry_month, return_dtype=pl.String)
                .replace(month_to_imm(as_dict=True, invert=True))
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
            hold_near=pl.col("hold_roll_cycle").str.extract_all("[F-Z]"),
            hold_far=pl.concat_str(
                pl.col("hold_roll_cycle").str.slice(1),
                pl.col("hold_roll_cycle").str.slice(0, 1),
            ).str.extract_all("[F-Z]"),
        )
        .explode("hold_near", "hold_far")
        .with_columns(
            hold_near_month=pl.col("hold_near")
            .replace(month_to_imm(as_dict=True, invert=True))
            .cast(pl.Int8),
            hold_far_month=pl.col("hold_far")
            .replace(month_to_imm(as_dict=True, invert=True))
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


def roll_calendar_time_grid(
    roll_config: pl.LazyFrame, start_date, end_date
) -> pl.LazyFrame:
    start_end = (
        roll_config.select(
            asof_date_offset=pl.lit(start_date)
            - (
                pl.lit(start_date)
                .dt.offset_by(pl.col("roll_offset"))
                .dt.offset_by(pl.col("expiry_offset"))
                .min()
            )
        )
        .select(
            start_date=pl.lit(start_date),
            end_date=pl.lit(end_date) + pl.col("asof_date_offset"),
        )
        .collect()
    )

    start_end = list(start_end.iter_rows())[0]
    start_end
    time_grid = pl.LazyFrame().time.range(start_end[0], start_end[1])

    return time_grid


def create_roll_calendars(
    roll_config: pl.LazyFrame, start_date, end_date
) -> pl.LazyFrame:
    time_grid = roll_calendar_time_grid(roll_config, start_date, end_date)
    return (
        time_grid.join(
            roll_config.select("asset", "expiry_offset", "roll_offset"), how="cross"
        )
        .with_columns(near_expiry_date=pl.lit(None).cast(pl.Date))
        .with_columns(
            near_expiry_date=(
                pl.when(pl.col("near_expiry_date").is_null()).then(
                    # approximate expriy date
                    pl.col("time")
                    .dt.month_start()
                    .dt.offset_by(pl.col("expiry_offset"))
                    .dt.date()
                )
            )
        )
        .unique(subset=["asset", "near_expiry_date"], keep="first", maintain_order=True)
        .with_columns(
            roll_date=(
                pl.col("near_expiry_date").dt.offset_by(pl.col("roll_offset")).dt.date()
            )
        )
        .with_columns(hold_near_month=pl.col("near_expiry_date").dt.month())
        .join(asset_contracts(roll_config), on=["asset", "hold_near_month"], how="left")
        .with_columns(pl.col("^hold_(near|far).*$").backward_fill().over("asset"))
        .with_columns(
            near_contract=pl.date(
                pl.col("near_expiry_date").dt.year(), pl.col("hold_near_month"), 1
            )
        )
        .with_columns(
            far_contract=pl.col("near_contract").dt.offset_by(
                pl.col("hold_far_month_offset")
            )
        )
        .with_columns(
            carry_contract=pl.col("near_contract").dt.offset_by(
                pl.col("carry_month_offset")
            )
        )
        .select(
            "roll_date",
            "asset",
            "near_contract",
            "near_expiry_date",
            "far_contract",
            "carry_contract",
        )
        .drop_nulls()
        .filter(pl.col("roll_date") <= end_date)
        .sort("asset", "roll_date")
    )

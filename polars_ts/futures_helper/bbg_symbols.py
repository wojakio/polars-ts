from datetime import datetime
import polars as pl

from .util import month_to_imm_dict

def bbg_symbol_universe(
    symbols: pl.LazyFrame,
    start_date: datetime,
    end_date: datetime
) -> pl.LazyFrame:
    
    imms = month_to_imm_dict()
    df = (
        pl.datetime_range(start_date, end_date, interval='1d', eager=True)
        .alias("time")
        .cast(pl.Date)
        .to_frame()
        .lazy()
        .join(symbols.select("asset", "symbol", "yellow_key").unique(), how="cross")
        .with_columns(imm=pl.col("time").dt.month().replace(imms).cast(pl.Categorical))
        .select(
            pl.col("time").dt.month_start(),
            "asset",
            "symbol",
            "imm",
            ticker=pl.concat_str([
                "symbol",
                pl.col("time").dt.month().replace(imms),
                pl.col("time").dt.year().cast(pl.String).str.slice(2),
                pl.lit(" "),
                pl.col("yellow_key")
            ]).cast(pl.Categorical)
        )
        .sort("symbol", "time")
        .unique(subset=["ticker"], keep="first", maintain_order=True)
    )

    return df.lazy()


def infer_bbg_contracts(
    roll_calendar: pl.LazyFrame,
    broker_meta: pl.LazyFrame
) -> pl.LazyFrame:

    contracts = (
        broker_meta
        .select(
            "asset",
            "symbol",
            yellow_key=pl.concat_str(pl.lit(" "), "yellow_key")
        )
    )

    df = (
        roll_calendar
        .join(contracts, on="asset", how="left")
        .calendar.date_to_imm_contract("near_contract", prefix="symbol", suffix="yellow_key")
        .calendar.date_to_imm_contract("far_contract", prefix="symbol", suffix="yellow_key")
        .calendar.date_to_imm_contract("carry_contract", prefix="symbol", suffix="yellow_key")
        .select(roll_calendar.columns)
    )

    return df
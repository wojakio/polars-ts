import polars as pl
from polars.type_aliases import IntoExpr

from ..utils import parse_into_expr
from .util import make_generic_contract


def generic_symbol_universe(
    symbols: pl.LazyFrame,
    start_date: IntoExpr,
    end_date: IntoExpr,
    date_pad_start: str = "0d",
    date_pad_end: str = "5y",
    date_step: str = "1d",
) -> pl.LazyFrame:
    start_date_expr = parse_into_expr(start_date, dtype=pl.Date)
    end_date_expr = parse_into_expr(end_date, dtype=pl.Date)

    df = (
        pl.datetime_range(
            start_date_expr.dt.offset_by(date_pad_start),
            end_date_expr.dt.offset_by(date_pad_end),
            interval=date_step,
            eager=True,
        )
        .alias("time")
        .cast(pl.Date)
        .to_frame()
        .lazy()
        .join(symbols.select("asset").unique(), how="cross")
        .select(
            pl.col("time").dt.month_start(),
            "asset",
            instrument_id=make_generic_contract("time"),
        )
        .sort("asset", "time")
        .unique(subset=["asset", "instrument_id"], keep="first", maintain_order=True)
    )

    return df.lazy()

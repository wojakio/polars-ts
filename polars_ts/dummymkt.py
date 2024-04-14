from typing import List, Union, Literal, Mapping, Optional

import polars as pl
from polars.type_aliases import IntoExpr

from .sf import SeriesFrame
from .utils import parse_into_expr

__NAMESPACE = "dummymkt"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class DummyMktFrame(SeriesFrame):
    def __init__(self, df: pl.LazyFrame):
        super().__init__(df)

    def fetch_instrument_prices(
        self,
        start_dt: IntoExpr,
        end_dt: IntoExpr,
        remove_weekends: bool = True
    ) -> pl.LazyFrame:
        start_dt_expr = parse_into_expr(start_dt, dtype=pl.Date)
        end_dt_expr = parse_into_expr(end_dt, dtype=pl.Date)
        hols = [6,7] if remove_weekends else []

        df = (
            self._df
            .select(
                time=pl.date_ranges(start_dt_expr, end_dt_expr),
                asset=pl.col("asset"),
                instrument_id=pl.col("instrument_id"),
            )
            .explode(pl.col("time"))
            .dummy.random_normal()
            .filter(pl.col("time").dt.weekday().is_in(hols).not_())
            .with_columns(pl.col("value").cum_sum().over("asset", "instrument_id"))
        )

        return df

    def fetch_roll_calendar_prices(
        self,
        roll_calendar: pl.LazyFrame,
        instrument_prices: pl.LazyFrame
    ) -> pl.LazyFrame:
        
        prices = instrument_prices.rename({"time": "roll_date"})

        df = (
            roll_calendar
            .join(prices.rename({"instrument_id": "near_contract", "value": "near_price"}), on=["roll_date", "asset", "near_contract"], how="left")
            .join(prices.rename({"instrument_id": "far_contract", "value": "far_price"}), on=["roll_date", "asset", "far_contract"], how="left")
            .join(prices.rename({"instrument_id": "carry_contract", "value": "carry_price"}), on=["roll_date", "asset", "carry_contract"], how="left")
        )

        return df
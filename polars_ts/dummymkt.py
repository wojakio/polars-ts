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
                ticker=pl.col("ticker"),
            )
            .explode(pl.col("time"))
            .dummy.random_normal()
            .filter(pl.col("time").dt.weekday().is_in(hols).not_())
            .with_columns(pl.col("value").cum_sum().over("asset", "ticker"))
        )

        return df
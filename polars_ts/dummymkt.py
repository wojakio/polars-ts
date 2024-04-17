import polars as pl
from polars.type_aliases import IntoExpr

from .sf import SeriesFrame
from .sf_helper import prepare_result
from .utils import parse_into_expr
from .dummymkt_helper import (
    impl_fetch_instrument_prices,
    impl_fetch_roll_calendar_prices,
)

__NAMESPACE = "dummymkt"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class DummyMktFrame(SeriesFrame):
    def __init__(self, df: pl.LazyFrame):
        super().__init__(df)

    def fetch_instrument_prices(
        self, start_dt: IntoExpr, end_dt: IntoExpr, remove_weekends: bool = True
    ) -> pl.LazyFrame:
        start_dt_expr = parse_into_expr(start_dt, dtype=pl.Date)
        end_dt_expr = parse_into_expr(end_dt, dtype=pl.Date)

        df = impl_fetch_instrument_prices(
            self._df, start_dt_expr, end_dt_expr, remove_weekends
        )
        return prepare_result(df)

    def fetch_roll_calendar_prices(
        self, roll_calendar: pl.LazyFrame, instrument_prices: pl.LazyFrame
    ) -> pl.LazyFrame:
        df = impl_fetch_roll_calendar_prices(roll_calendar, instrument_prices)
        return prepare_result(df)

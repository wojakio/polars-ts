from typing import Generic
import polars as pl
from polars.type_aliases import IntoExpr

from .sf import SeriesFrame
from .sf_helper import prepare_result
from .utils import parse_into_expr
from .dummymkt_helper import (
    impl_fetch_instrument_prices,
    impl_fetch_roll_calendar_prices,
)

from .types import FrameType

__NAMESPACE = "dummymkt"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class DummyMktFrame(SeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType):
        super().__init__(df)

    def fetch_instrument_prices(
        self, start_dt: IntoExpr, end_dt: IntoExpr, remove_weekends: bool = True
    ) -> FrameType:
        start_dt_expr = parse_into_expr(start_dt, dtype=pl.Date)
        end_dt_expr = parse_into_expr(end_dt, dtype=pl.Date)

        df = impl_fetch_instrument_prices(
            self._df, start_dt_expr, end_dt_expr, remove_weekends
        )
        return prepare_result(df)

    def fetch_roll_calendar_prices(
        self,
        roll_calendar: FrameType,
        instrument_prices: FrameType,
        stitch_lookback_interval: str = "15d",
    ) -> FrameType:
        df = impl_fetch_roll_calendar_prices(
            roll_calendar, instrument_prices, stitch_lookback_interval
        )
        return prepare_result(df)

from datetime import datetime
from typing import Optional, Generic

import polars as pl
from polars.type_aliases import IntoExpr

from .sf import SeriesFrame
from .sf_helper import prepare_result
from .types import FrameType
from .time_helper import impl_range, impl_ranges
from .utils import parse_into_expr

__NAMESPACE = "time"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class TimeFrame(SeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType) -> None:
        super().__init__(df)

    def range(
        self,
        t0: datetime,
        t1: Optional[datetime] = None,
        offset: str = "0d",
        interval: str = "1d",
    ) -> FrameType:
        df = impl_range(self._df, t0, t1, offset, interval)

        return prepare_result(df)

    def ranges(
        self,
        start_date: IntoExpr,
        end_date: IntoExpr,
        skip_dates: IntoExpr = pl.lit([]),
        iso_weekends: IntoExpr = pl.lit([1, 2, 3, 4, 5]),
        out_name: str = "ranges",
    ) -> FrameType:
        start_date = parse_into_expr(start_date)
        end_date = parse_into_expr(end_date)
        skip_dates = parse_into_expr(skip_dates)
        iso_weekends = parse_into_expr(iso_weekends, dtype=pl.List(pl.Int8))

        df = impl_ranges(
            self._df, start_date, end_date, skip_dates, iso_weekends, out_name
        )
        return prepare_result(df)

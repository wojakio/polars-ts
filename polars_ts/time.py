from datetime import datetime
from typing import Optional

import polars as pl

from .sf import SeriesFrame

__NAMESPACE = "time"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class TimeFrame(SeriesFrame):
    def __init__(self, df: pl.LazyFrame) -> None:
        self._df = df

    def range(
        self,
        t0: datetime,
        t1: Optional[datetime] = None,
        offset: str = "0d",
        interval: str = "1d",
    ) -> pl.LazyFrame:
        self._df = self._df.with_columns(
            time=(
                pl.lit([t0, t1])
                .list.concat(pl.lit(t0).dt.offset_by(offset))
                .list.drop_nulls()
                .list.unique()
                .list.sort()
            )
        )

        return self.expand(interval)

    def expand(self, interval: str = "1d") -> pl.Series:
        # pre: every row has a time: list[date] column
        # xdt doesnt support date_ranges?
        # fn = xdt.date_range if interval.endswith('bd') else pl.date_ranges
        fn = pl.date_ranges

        self._df = self._df.with_columns(
            fn(pl.col("time").list.min(), pl.col("time").list.max(), interval)
        ).explode("time")

        return self._df

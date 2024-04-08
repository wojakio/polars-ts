from typing import List, Literal, Mapping

import polars as pl

from .tsf import TimeSeriesFrame

__NAMESPACE = "rs"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class ResampleFrame(TimeSeriesFrame):
    def __init__(self, df: pl.LazyFrame):
        super().__init__(df)

    def to_ohlc(
        self,
        period: str,
        *,
        partition: Mapping[Literal["by", "but"], List[str]] = None,
        value_col: str = "value",
    ) -> pl.LazyFrame:
        split = self.auto_partition(partition)

        self._df = (
            self._df.sort("time")
            .group_by_dynamic("time", every=period, group_by=split)
            .agg(
                pl.first(value_col).alias("open"),
                pl.max(value_col).alias("high"),
                pl.min(value_col).alias("low"),
                pl.last(value_col).alias("close"),
            )
        )

        return self.result_df

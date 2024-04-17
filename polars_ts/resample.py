from typing import List, Literal, Mapping, Optional, Union

import polars as pl

from .resample_helper import *
from .tsf import TimeSeriesFrame
from .types import *

__NAMESPACE = "rs"

@pl.api.register_lazyframe_namespace(__NAMESPACE)
class ResampleFrame(TimeSeriesFrame):
    def __init__(self, df: pl.LazyFrame):
        super().__init__(df)

    def to_ohlc(
        self,
        period: str,
        *,
        partition: PartitionType = None,
        value_col: str = "value",
    ) -> pl.LazyFrame:
        partition = impl_auto_partition(self._df, partition)

        df = (
            self._df.sort("time")
            .group_by_dynamic("time", every=period, group_by=partition)
            .agg(
                pl.first(value_col).alias("open"),
                pl.max(value_col).alias("high"),
                pl.min(value_col).alias("low"),
                pl.last(value_col).alias("close"),
            )
        )

        return prepare_result(df)


    def align_to_time(
        self,
        time_axis: pl.Series,
        partition: PartitionType = None,
        closed: IntervalType = 'left',
        fill_strategy: FillStrategyType = 'forward',
        fill_sentinel: Union[float, int]=0.,
    ) -> pl.LazyFrame:

        df = impl_align_to_time(
            self._df,
            time_axis,
            partition,
            closed,
            fill_strategy,
            fill_sentinel
        )

        return prepare_result(df)
        

    def resample_categories(
        self,
        time_axis: pl.Series,
        partition: PartitionType = None,
        closed: IntervalType = 'left'
    ) -> pl.LazyFrame:

        df = impl_resample_categories(
            self._df,
            time_axis,
            partition,
            closed
        )

        return prepare_result(df)


    def align_values(
        self,
        rhs: pl.LazyFrame,
        partition: PartitionType = None,
        retain_values: Literal['lhs', 'rhs', 'both'] = 'lhs',
        fill_strategy: FillStrategyType = 'forward',
        fill_sentinel: Union[float, int] = 0.
    ) -> pl.LazyFrame:

        df = impl_align_values(
            self._df,
            rhs,
            partition,
            retain_values,
            fill_strategy,
            fill_sentinel
        )

        return prepare_result(df)
   
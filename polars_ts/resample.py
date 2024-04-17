from typing import Literal, Union, Generic

import polars as pl

from .sf_helper import prepare_result

from .resample_helper import (
    impl_align_to_time,
    impl_align_values,
    impl_resample_categories,
)

from .grouper import Grouper
from .tsf import TimeSeriesFrame
from .types import (
    IntervalType,
    FillStrategyType,
    FrameType,
)

__NAMESPACE = "rs"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class ResampleFrame(TimeSeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType):
        super().__init__(df)

    def to_ohlc(
        self,
        period: str,
        *,
        partition: Grouper = Grouper(),
        value_col: str = "value",
    ) -> FrameType:
        grouper_cols = partition.apply(self._df)

        df = (
            self._df.sort("time")
            .group_by_dynamic("time", every=period, group_by=grouper_cols)
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
        partition: Grouper = Grouper(),
        closed: IntervalType = "left",
        fill_strategy: FillStrategyType = "forward",
        fill_sentinel: Union[float, int] = 0.0,
    ) -> FrameType:
        df = impl_align_to_time(
            self._df, time_axis, partition, closed, fill_strategy, fill_sentinel
        )

        return prepare_result(df)

    def resample_categories(
        self,
        time_axis: pl.Series,
        partition: Grouper = Grouper(),
        closed: IntervalType = "left",
    ) -> FrameType:
        df = impl_resample_categories(self._df, time_axis, partition, closed)

        return prepare_result(df)

    def align_values(
        self,
        rhs: FrameType,
        partition: Grouper = Grouper(),
        retain_values: Literal["lhs", "rhs", "both"] = "lhs",
        fill_strategy: FillStrategyType = "forward",
        fill_sentinel: Union[float, int] = 0.0,
    ) -> FrameType:
        df = impl_align_values(
            self._df, rhs, partition, retain_values, fill_strategy, fill_sentinel
        )

        return prepare_result(df)

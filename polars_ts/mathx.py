from typing import Generic

import polars as pl

from .grouper import Grouper

from .sf import SeriesFrame
from .sf_helper import prepare_result

from .mathx_helper import (
    impl_diff,
    impl_cum_sum,
    impl_ewm_mean,
    impl_shift,
)

from .types import FrameType

__NAMESPACE = "mathx"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class MathxFrame(SeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType):
        super().__init__(df)

    def diff(
        self,
        params: FrameType,
        partition: Grouper = Grouper.by_all_and("n"),
    ) -> FrameType:
        df = impl_diff(self._df, params, partition)
        return prepare_result(df)

    def cum_sum(self, partition: Grouper = Grouper.by_all()) -> FrameType:
        df = impl_cum_sum(self._df, partition)
        return prepare_result(df)

    def shift(
        self,
        params: FrameType,
        partition: Grouper = Grouper.by_all_and("n"),
    ) -> FrameType:
        df = impl_shift(self._df, params, partition)
        return prepare_result(df)

    def ewm_mean(
        self,
        params: FrameType,
        partition: Grouper = Grouper.by_all_and("alpha"),
    ) -> FrameType:
        df = impl_ewm_mean(self._df, params, partition)
        return prepare_result(df)

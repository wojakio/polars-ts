from typing import Any, Generic, Optional

import polars as pl

from .grouper import Grouper

from .sf import SeriesFrame, impl_handle_null
from .sf_helper import prepare_params, prepare_result

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
        partition: Grouper = Grouper.by_all_and("n"),
        *,
        n: int = 1,
        method: str = 'arithmetic',
        null_strategy: str = 'ignore',
        null_param_1: Any = None,
        params: Optional[FrameType] = None,
    ) -> FrameType:
        params = prepare_params(
            self._df,
            params,
            n=n,
            method=method,
            null_strategy=null_strategy,
            null_param_1=null_param_1,
        )
        df = impl_diff(self._df, partition, params)
        # df = impl_handle_null(df, partition, params)
        return prepare_result(df)

    def cum_sum(self, partition: Grouper = Grouper.by_all()) -> FrameType:
        df = impl_cum_sum(self._df, partition)
        return prepare_result(df)

    def shift(
        self,
        partition: Grouper = Grouper.by_all_and("n"),
        *,
        n: int = 1,
        null_strategy: str = 'ignore',
        null_param_1: Any = None,
        params: Optional[FrameType] = None,
    ) -> FrameType:
        params = prepare_params(
            self._df,
            params,
            n=n,
            null_strategy=null_strategy,
            null_param_1=null_param_1,
        )

        df = impl_shift(self._df, partition, params)
        return prepare_result(df)


    def ewm_mean(
        self,
        partition: Grouper = Grouper.by_all_and("alpha"),
        *,
        alpha: float = 0.5,
        min_periods: int = 0,
        adjust: bool = False,
        null_strategy: str = 'ignore',
        null_param_1: Any = None,
        params: Optional[FrameType] = None,
    ) -> FrameType:
        params = prepare_params(
            self._df,
            params,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            null_strategy=null_strategy,
            null_param_1=null_param_1,
        )

        df = impl_ewm_mean(self._df, partition, params)
        return prepare_result(df)

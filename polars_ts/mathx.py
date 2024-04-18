from typing import Literal, Generic

import polars as pl

from .grouper import Grouper

from .sf import SeriesFrame
from .sf_helper import prepare_result

from .mathx_helper import impl_diff

from .types import FrameType, NullBehaviorType

__NAMESPACE = "mathx"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class MathxFrame(SeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType):
        super().__init__(df)

    def diff(
        self,
        k: int,
        method: Literal["arithmetic", "fractional", "geometric"] = "arithmetic",
        partition: Grouper = Grouper(),
        null_behavior: NullBehaviorType = "drop",
    ) -> FrameType:
        df = impl_diff(self._df, k, method, partition, null_behavior)

        return prepare_result(df)

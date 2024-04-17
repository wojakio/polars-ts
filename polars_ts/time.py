from datetime import datetime
from typing import Optional, Generic

import polars as pl

from .sf import SeriesFrame
from .sf_helper import prepare_result
from .types import FrameType
from .time_helper import impl_range

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

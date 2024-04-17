from typing import Generic
import polars as pl

from .sf import SeriesFrame
from .types import FrameType

__NAMESPACE = "tsf"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class TimeSeriesFrame(SeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType):
        if "time" not in df.columns:
            raise ValueError("Missing column: ", "time")

        super().__init__(df)

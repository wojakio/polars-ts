from typing import Generic
import polars as pl

from .sf import SeriesFrame
from .types import FrameType

__NAMESPACE = "io"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class IoFrame(SeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType) -> None:
        super().__init__(df)

    def read_csv(self, filename: str, **kwargs) -> pl.LazyFrame:
        df = pl.scan_csv(filename, comment_prefix="//", **kwargs).with_columns(
            pl.col(pl.String).cast(pl.Categorical)
        )

        return df

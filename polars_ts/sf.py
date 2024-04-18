from typing import Generic

import polars as pl

from .sf_helper import RESERVED_ALL_GRP
from .types import FrameType

__NAMESPACE = "sf"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class SeriesFrame(Generic[FrameType]):
    def __init__(self, df: FrameType):
        self._df: FrameType = df.with_columns(
            pl.lit(0, pl.Boolean).cast(pl.Categorical).alias(RESERVED_ALL_GRP)
        )

from typing import Generic

import polars as pl
from polars.type_aliases import JoinStrategy

from .grouper import Grouper
from .sf_helper import RESERVED_ALL_GRP, impl_join, prepare_result, impl_unique
from .types import FrameType

__NAMESPACE = "sf"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class SeriesFrame(Generic[FrameType]):
    def __init__(self, df: FrameType):
        self._df: FrameType = df.with_columns(
            pl.lit("_placeholder_").cast(pl.Categorical).alias(RESERVED_ALL_GRP)
        )

    def join(
        self,
        other: FrameType,
        grouper: Grouper = Grouper().by_common_including_time(),
        how: JoinStrategy = "inner",
    ) -> FrameType:
        df = impl_join(self._df, other, grouper, how)
        return prepare_result(df)

    def unique(self, grouper: Grouper = Grouper().by_time_and_all()) -> FrameType:
        df = impl_unique(self._df, grouper)
        return prepare_result(df)

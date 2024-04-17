from typing import Iterable, List, Generic

import polars as pl

from .sf_helper import impl_categories, RESERVED_ALL_GRP
from .types import FrameType

__NAMESPACE = "sf"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class SeriesFrame(Generic[FrameType]):
    def __init__(self, df: FrameType):
        self._df: FrameType = df.with_columns(
            pl.lit(0, pl.Boolean).cast(pl.Categorical).alias(RESERVED_ALL_GRP)
        )

    def _value_types(self):
        return pl.NUMERIC_DTYPES

    def category_names(self) -> List[str]:
        return impl_categories(self._df).columns

    def value_names(self) -> Iterable[str]:
        return set(self._df.columns).difference(self.category_names())

    def values(self) -> FrameType:
        return self._df.select(self.value_names())

    def categories(self) -> FrameType:
        return impl_categories(self._df)

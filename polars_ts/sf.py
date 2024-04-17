from typing import Iterable, List, Mapping, Literal, Union

import polars as pl

from .sf_helper import *

__NAMESPACE = "sf"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class SeriesFrame:

    def __init__(self, df: pl.LazyFrame):
        self._df = df.with_columns(
            pl.lit(0, pl.Boolean).cast(pl.Categorical).alias(RESERVED_ALL_GRP)
        )

    def _value_types(self) -> List[pl.DataType]:
        return sorted(pl.NUMERIC_DTYPES)

    def category_names(self) -> List[str]:
        return impl_categories(self._df).columns

    def value_names(self) -> Iterable[str]:
        return set(self._df.columns).difference(self.category_names())

    def values(self) -> pl.LazyFrame:
        return self._df.select(self.value_names())

    def categories(self) -> pl.LazyFrame:
        return impl_categories(self._df)

    def ht(self, nrows=3) -> pl.LazyFrame:
        return pl.concat([self._df.head(nrows), self._df.tail(nrows)]).unique()

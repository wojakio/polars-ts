from typing import Iterable, List, Mapping, Literal

import polars as pl

__NAMESPACE = "sf"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class SeriesFrame:
    _RESERVED_COL_PREFIX = "##@_"
    _RESERVED_COL_REGEX = "^##@_.*$"
    _RESERVED_ALL_GRP = f"{_RESERVED_COL_PREFIX}_GRP_ALL"
    _RESERVED_ROW_IDX = f"{_RESERVED_COL_PREFIX}_INDEX"

    def __init__(self, df: pl.LazyFrame):
        self._df = df.with_columns(pl.lit(0, pl.Boolean).alias(self._RESERVED_ALL_GRP))

    @property
    def result_df(self) -> pl.LazyFrame:
        return self._df.select(pl.exclude(self._RESERVED_COL_REGEX))

    def _value_types(self):
        return pl.NUMERIC_DTYPES

    def category_names(self) -> List[str]:
        return self.categories().columns

    def value_names(self) -> Iterable[str]:
        return set(self._df.columns).difference(self.category_names())

    def values(self) -> pl.LazyFrame:
        return self._df.select(self.value_names())

    def categories(self) -> pl.LazyFrame:
        return self._df.select(pl.col(pl.Categorical, pl.Enum))

    def ht(self, nrows=3) -> pl.LazyFrame:
        if len(self._df) <= (nrows + nrows):
            return self._df

        return pl.concat([self._df.head(nrows), self._df.tail(nrows)])

    def common_category_names(self, rhs: pl.LazyFrame) -> List[str]:
        return sorted(set(self.category_names()).intersection(rhs.sf.category_names()))

    def auto_partition(
        self,
        partition: Mapping[Literal["by", "but"], List[str]],
    ) -> List[str]:
        cols = set(self.categories().columns).difference(["time"])
        if partition is not None:
            if len(set(partition.keys()).intersection(["by", "but"])) > 1:
                raise ValueError("Must specify 'by' xor 'but'")

            if "by" in partition:
                by = partition["by"]
                cols = cols.intersection(by)

            if "but" in partition:
                but = partition["but"]
                cols = cols.difference(but)

            if len(cols) == 0:
                raise ValueError(f"Invalid partition spec. {partition}")

        return sorted(cols)

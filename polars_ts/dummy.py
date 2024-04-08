from typing import List, Union, Literal, Mapping

import polars as pl

from .sf import SeriesFrame
from .expr.random import (
    random_normal as rand_normal,
    random_uniform as rand_uniform,
    wyhash,
)

__NAMESPACE = "dummy"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class DummyFrame(SeriesFrame):
    def __init__(self, df: pl.LazyFrame):
        super().__init__(df)

    def random_category_subgroups(
        self,
        prefix: pl.Expr,
        sizes: List[int],
        *,
        max_num_subgroups: int = None,
        seed: int = 42,
        partition: Mapping[Literal["by", "but"], List[str]] = None,
        out: str = "subgroup",
    ) -> pl.LazyFrame:
        split = self.auto_partition(partition)

        if max_num_subgroups is None:
            max_num_subgroups = (
                self._df.select(1 + (pl.len() // max(sizes))).collect().item(0, 0)
            )

        prefix = pl.col(prefix) if isinstance(prefix, str) else prefix

        idxs = (
            pl.Series(sizes, dtype=pl.UInt64)
            .sample(n=max_num_subgroups, with_replacement=True, seed=seed)
            .cum_sum()
        )

        self._df = self._df.with_row_index(self._RESERVED_ROW_IDX).with_columns(
            pl.concat_str(
                prefix,
                pl.when(pl.col(self._RESERVED_ROW_IDX).is_in(idxs))
                .then(1)
                .otherwise(0)
                .cum_sum()
                .cast(pl.String),
            )
            .over(split)
            .cast(pl.Categorical)
            .alias(out)
        )

        return self.result_df

    def random_uniform(
        self,
        lower: Union[pl.Expr, float] = 0.0,
        upper: Union[pl.Expr, float] = 1.0,
        *,
        partition: Mapping[Literal["by", "but"], List[str]] = None,
        out: str = "value",
    ) -> pl.LazyFrame:
        split = self.auto_partition(partition)
        self._df = self._df.with_columns(
            rand_uniform(lower, upper, seed=wyhash(pl.concat_str(split).first()))
            .over(split)
            .alias(out)
        )

        # random nulls
        # dtype

        return self.result_df

    def random_normal(
        self,
        mu: Union[pl.Expr, float] = 0.0,
        sigma: Union[pl.Expr, float] = 1.0,
        *,
        partition: Mapping[Literal["by", "but"], List[str]] = None,
        out: str = "value",
    ) -> pl.LazyFrame:
        split = self.auto_partition(partition)
        self._df = self._df.with_columns(
            rand_normal(mu, sigma, seed=wyhash(pl.concat_str(split).first()))
            .over(split)
            .alias(out)
        )

        return self.result_df

    def enum(
        self, names: Union[List[str], int], *, out: str = "category", prefix="ENUM_"
    ) -> pl.LazyFrame:
        if isinstance(names, int):
            names_expr = (
                pl.int_ranges(0, 100, eager=True)
                .cast(pl.List(pl.Utf8))
                .list.eval(pl.element().str.replace("^", prefix))
                .alias(out)
            )
            enum_dtype = pl.Enum(names_expr.explode().to_list())

        else:
            names_expr = pl.lit(names).alias(out)
            enum_dtype = pl.Enum(names)

        # random nulls

        self._df = (
            self._df.with_columns(names_expr)
            .explode(out)
            .with_columns(pl.col(out).cast(enum_dtype))
        )
        return self.result_df

    def correlate(
        self,
        col_a,
        col_b,
        type: Literal[
            "additive", "multiplicative", "shift", "exponent", "average", "none"
        ],
    ):
        # correlate with col
        #   correlate type: noise, lead/lag, moving avg, non-linear

        pass

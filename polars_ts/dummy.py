from typing import List, Union, Literal, Optional, Generic

import polars as pl

from polars.type_aliases import IntoExpr
from .grouper import Grouper

from .sf import SeriesFrame
from .sf_helper import prepare_result
from .dummy_helper import (
    impl_random_category_subgroups,
    impl_random_normal,
    impl_random_uniform,
    impl_enum,
)

from .utils import parse_into_expr
from .types import FrameType

__NAMESPACE = "dummy"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class DummyFrame(SeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType):
        super().__init__(df)

    def random_category_subgroups(
        self,
        prefix: IntoExpr,
        sizes: List[int],
        *,
        max_num_subgroups: Optional[int] = None,
        seed: int = 42,
        partition: Grouper = Grouper.by_all(),
        out: str = "subgroup",
    ) -> FrameType:
        prefix = parse_into_expr(prefix)

        if max_num_subgroups is None:
            size_estimate = self._df.select(1 + (pl.len() // max(sizes)))
            if isinstance(size_estimate, pl.LazyFrame):
                size_estimate = size_estimate.collect()

            max_num_subgroups = size_estimate.item(0, 0)

        df = impl_random_category_subgroups(
            self._df, prefix, sizes, max_num_subgroups, seed, partition, out
        )

        return prepare_result(df)

    def random_uniform(
        self,
        lower: Union[pl.Expr, float] = 0.0,
        upper: Union[pl.Expr, float] = 1.0,
        *,
        partition: Grouper = Grouper.by_all(),
        out: str = "value",
    ) -> FrameType:
        df = impl_random_uniform(self._df, lower, upper, partition, out)

        return prepare_result(df)

    def random_normal(
        self,
        mu: Union[pl.Expr, float] = 0.0,
        sigma: Union[pl.Expr, float] = 1.0,
        *,
        partition: Grouper = Grouper.by_all(),
        out: str = "value",
    ) -> FrameType:
        df = impl_random_normal(self._df, partition, out, mu, sigma)

        return prepare_result(df)

    def enum(
        self, names: Union[List[str], int], *, out: str = "category", prefix="ENUM_"
    ) -> FrameType:
        df = impl_enum(self._df, names, out, prefix)
        return prepare_result(df)

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

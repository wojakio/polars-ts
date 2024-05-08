from typing import Union

import polars as pl

from ..sf_helper import RESERVED_ROW_IDX

from ..types import FrameType
from ..grouper import Grouper
from typing import List, Literal

from ..expr.random import (
    random_normal as rand_normal,
    random_uniform as rand_uniform,
    wyhash,
)


def impl_random_category_subgroups(
    df: FrameType,
    prefix: Union[str, pl.Expr],
    sizes: List[int],
    max_num_subgroups: int,
    seed: int,
    partition: Grouper,
    out: str,
) -> FrameType:
    grouper_cols = partition.apply(df)
    idxs = (
        pl.Series(sizes, dtype=pl.UInt64)
        .sample(n=max_num_subgroups, with_replacement=True, seed=seed)
        .cum_sum()
    )

    result = df.with_row_index(RESERVED_ROW_IDX).with_columns(
        pl.concat_str(
            prefix,
            pl.when(pl.col(RESERVED_ROW_IDX).is_in(idxs))
            .then(1)
            .otherwise(0)
            .cum_sum()
            .cast(pl.String),
        )
        .over(grouper_cols)
        .cast(pl.Categorical)
        .alias(out)
    )

    return result


def impl_random_uniform(
    df: FrameType,
    lower: Union[pl.Expr, float],
    upper: Union[pl.Expr, float],
    partition: Grouper,
    out: str,
) -> FrameType:
    grouper_cols = partition.apply(df)
    result = df.with_columns(
        rand_uniform(lower, upper, seed=wyhash(pl.concat_str(grouper_cols).first()))
        .over(grouper_cols)
        .alias(out)
    )

    # random nulls
    # dtype

    return result


def impl_random_normal(
    df: FrameType,
    partition: Grouper,
    out: str,
    mu: Union[pl.Expr, float],
    sigma: Union[pl.Expr, float],
) -> FrameType:
    grouper_cols = partition.apply(df)
    result = df.with_columns(
        rand_normal(mu, sigma, seed=wyhash(pl.concat_str(grouper_cols).first()))
        .over(grouper_cols)
        .alias(out)
    )

    return result


def impl_enum(
    df: FrameType, names: Union[List[str], int], out: str, prefix: str
) -> FrameType:
    if isinstance(names, int):
        names_expr = (
            pl.int_ranges(0, 100)
            .cast(pl.List(pl.Utf8))
            .list.eval(pl.element().str.replace("^", prefix))
            .alias(out)
        )
        enum_dtype = pl.Enum(
            pl.DataFrame().with_columns(names_expr.explode()).to_series().to_list()
        )

    else:
        names_expr = pl.lit(names).alias(out)
        enum_dtype = pl.Enum(names)

    # random nulls

    result = (
        df.with_columns(names_expr)
        .explode(out)
        .with_columns(pl.col(out).cast(enum_dtype))
    )
    return result


def impl_correlate(
    df: FrameType,
    col_a,
    col_b,
    type: Literal["additive", "multiplicative", "shift", "exponent", "average", "none"],
):
    # correlate with col
    #   correlate type: noise, lead/lag, moving avg, non-linear

    pass

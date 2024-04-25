import math
from typing import Literal

import polars as pl

from ..sf_helper import impl_fill_null
from ..grouper import Grouper

from ..types import FrameType, NullStrategyType, SentinelNumeric


def impl_diff(
    df: FrameType,
    k: int,
    method: Literal["arithmetic", "fractional", "geometric"],
    partition: Grouper,
    null_strategy: NullStrategyType,
    null_sentinel: SentinelNumeric,
) -> FrameType:
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df)

    if method == "arithmetic":
        result_expr = pl.col(numeric_cols).diff(k).over(grouper_cols)
    elif method == "fractional":
        result_expr = pl.col(numeric_cols).pct_change(k).over(grouper_cols)
    elif method == "geometric":
        result_expr = (pl.col(numeric_cols) / (pl.col(numeric_cols).shift(k))  - 1).over(grouper_cols)
    else:
        raise ValueError(f"Unexpected diff method: {method}")

    result = df.with_columns(result_expr)
    result = impl_fill_null(result, null_strategy, null_sentinel, partition)

    return result


def impl_cum_sum(df: FrameType, partition: Grouper) -> FrameType:
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df)

    result = df.with_columns(pl.col(numeric_cols).cum_sum().over(grouper_cols))

    return result


def impl_shift(
    df: FrameType,
    k: int,
    null_strategy: NullStrategyType,
    null_sentinel: SentinelNumeric,
    partition: Grouper,
) -> FrameType:
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df)

    result = df.with_columns(pl.col(numeric_cols).shift(k).over(grouper_cols))
    result = impl_fill_null(result, null_strategy, null_sentinel, partition)

    return result


def impl_ewm_mean(
    df: FrameType,
    half_life: float,
    partition: Grouper,
) -> FrameType:
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df)

    result = df.with_columns(
        pl.col(numeric_cols)
        .ewm_mean(half_life=half_life, ignore_nulls=True, adjust=True)
        .over(grouper_cols)
    )
    return result


def half_life_to_alpha(lam: float) -> float:
    return -1 / math.log(lam)

from typing import Literal

import polars as pl

from ..sf_helper import impl_apply_null_strategy
from ..grouper import Grouper

from ..types import FrameType, NullStrategyType, SentinelNumeric


def impl_diff(
    df: FrameType,
    k: int,
    method: Literal["arithmetic", "fractional", "geometric"],
    partition: Grouper,
    null_strategy: NullStrategyType,
    null_sentinel_numeric: SentinelNumeric,
) -> FrameType:
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df)

    result = df.with_columns(pl.col(numeric_cols).diff(k).over(grouper_cols))
    result = impl_apply_null_strategy(
        result, null_strategy, null_sentinel_numeric, partition
    )

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
    null_sentinel_numeric: SentinelNumeric,
    partition: Grouper,
) -> FrameType:
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df)

    result = df.with_columns(pl.col(numeric_cols).shift(k).over(grouper_cols))
    result = impl_apply_null_strategy(
        result, null_strategy, null_sentinel_numeric, partition
    )

    return result


def impl_ewm_mean(
    df: FrameType,
    alpha: int,
    partition: Grouper,
) -> FrameType:
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df)

    result = df.with_columns(
        pl.col(numeric_cols).ewm_mean(alpha=alpha).over(grouper_cols)
    )
    return result

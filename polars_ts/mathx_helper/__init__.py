from typing import Literal

import polars as pl

from ..expr.mathx import ewm_custom, shift_custom
from ..sf_helper import impl_fill_null
from ..grouper import Grouper
from ..with_params import WithParams

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
        result_expr = (pl.col(numeric_cols) / (pl.col(numeric_cols).shift(k))).over(
            grouper_cols
        )
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
    params: FrameType,
    partition: Grouper,
) -> FrameType:
    p = WithParams().optional(shift=pl.Int64).defaults(shift=1)

    df, result_cols = p.apply(df, params)
    grouper_cols = partition.apply(df)

    result = df.with_columns(
        shift_custom(
            pl.col(partition.numerics(df)),
            "shift",
        ).over(grouper_cols)
    ).select(result_cols)

    return result


def impl_ewm_mean(
    df: FrameType,
    params: FrameType,
    partition: Grouper,
) -> FrameType:
    p = (
        WithParams(alpha=pl.Float64)
        .optional(min_periods=pl.Int64, adjust=pl.Boolean)
        .defaults(min_periods=0, adjust=False)
    )

    df, result_cols = p.apply(df, params)
    grouper_cols = partition.apply(df)

    result = df.with_columns(
        ewm_custom(
            pl.col(partition.numerics(df)),
            "alpha",
            "min_periods",
            "adjust",
        ).over(grouper_cols)
    ).select(result_cols)

    return result

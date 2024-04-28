from typing import Literal

import polars as pl
from polars.type_aliases import JoinStrategy

from ..expr.mathx import ewm_custom, shift_custom
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


def impl_shift_config(
    df: FrameType,
    params: FrameType,
    partition: Grouper,
) -> FrameType:
    result_schema = (
        pl.Series(values=df.columns + Grouper.categories(params, include_time=False))
        .unique()
        .to_list()
    )

    default_params = params.select(
        pl.Series("shift", [1]),
    )

    missing_params = set(default_params.columns).difference(params.columns)
    if len(missing_params) > 0:
        params = params.join(default_params.select(missing_params), how="cross")

    common_cols = Grouper.common_categories(df, params)
    join_type: JoinStrategy = "cross" if len(common_cols) == 0 else "left"
    df = df.join(params, on=common_cols, how=join_type)

    grouper_cols = partition.apply(df)

    result = df.with_columns(
        shift_custom(
            pl.col(partition.numerics(df)),
            "shift",
        ).over(grouper_cols)
    ).select(result_schema)

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


def impl_ewm_mean_config(
    df: FrameType,
    params: FrameType,
    partition: Grouper,
) -> FrameType:
    result_schema = (
        pl.Series(values=df.columns + Grouper.categories(params, include_time=False))
        .unique()
        .to_list()
    )

    default_params = params.select(
        pl.Series("min_periods", [0]),
        pl.Series("adjust", [False]),
    )

    missing_params = set(default_params.columns).difference(params.columns)
    if len(missing_params) > 0:
        params = params.join(default_params.select(missing_params), how="cross")

    common_cols = Grouper.common_categories(df, params)
    join_type: JoinStrategy = "cross" if len(common_cols) == 0 else "left"
    df = df.join(params, on=common_cols, how=join_type)

    grouper_cols = partition.apply(df)

    result = df.with_columns(
        ewm_custom(
            pl.col(partition.numerics(df)),
            "alpha",
            "min_periods",
            "adjust",
        ).over(grouper_cols)
    ).select(result_schema)

    return result

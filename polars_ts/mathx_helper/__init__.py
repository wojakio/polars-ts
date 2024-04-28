import polars as pl

from ..expr.mathx import diff_custom, ewm_custom, shift_custom

# from ..sf_helper import impl_fill_null
from ..grouper import Grouper
from ..with_params import WithParams

from ..types import FrameType


def impl_diff(
    df: FrameType,
    params: FrameType,
    partition: Grouper,
) -> FrameType:
    p = (
        WithParams()
        .optional(
            n=pl.Int64,
            method=pl.Categorical,
            null_strategy=pl.String,
            null_sentinel=pl.Float64,
        )
        .defaults(n=1, method="arithmetic", null_strategy="drop", null_sentinel=0.0)
    )

    df, result_cols = p.apply(df, params)

    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df, exclude=p.names())

    result = df.with_columns(
        diff_custom(
            pl.col(numeric_cols),
            "n",
            "method",
        ).over(grouper_cols)
    ).select(result_cols)

    # .select(handle_nulls(pl.col(numeric_cols), strategy, sentinel))

    # result = impl_fill_null(result, null_strategy, null_sentinel, partition)

    return result


def impl_cum_sum(df: FrameType, partition: Grouper) -> FrameType:
    p = WithParams()
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df, exclude=p.names())

    result = df.with_columns(pl.col(numeric_cols).cum_sum().over(grouper_cols))

    return result


def impl_shift(
    df: FrameType,
    params: FrameType,
    partition: Grouper,
) -> FrameType:
    p = WithParams(n=pl.Int64)

    df, result_cols = p.apply(df, params)
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df, exclude=p.names())

    result = df.with_columns(
        shift_custom(
            pl.col(numeric_cols),
            "n",
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
    numeric_cols = partition.numerics(df, exclude=p.names())

    result = df.with_columns(
        ewm_custom(
            pl.col(numeric_cols),
            "alpha",
            "min_periods",
            "adjust",
        ).over(grouper_cols)
    ).select(result_cols)

    return result

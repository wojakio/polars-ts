import polars as pl

from ..expr.mathx import diff_custom, ewm_custom, shift_custom

# from ..sf_helper import impl_fill_null
from ..grouper import Grouper
from ..param_schema import ParamSchema

from ..types import FrameType


def impl_diff(
    df: FrameType,
    partition: Grouper,
    params: FrameType,
) -> FrameType:
    p = (
        ParamSchema()
        .optional(
            n=pl.Int64,
            method=pl.Categorical,
            null_strategy=pl.Categorical,
            null_param_1=pl.Float64,
        )
        .defaults(n=1, method="arithmetic", null_strategy="ignore", null_param_1=None)
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

    return result


def impl_cum_sum(df: FrameType, partition: Grouper) -> FrameType:
    p = ParamSchema()
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df, exclude=p.names())

    result = df.with_columns(pl.col(numeric_cols).cum_sum().over(grouper_cols))

    return result


def impl_shift(
    df: FrameType,
    partition: Grouper,
    params: FrameType,
) -> FrameType:
    p = (
        ParamSchema(n=pl.Int64)
        .optional(
            null_strategy=pl.Categorical,
            null_param_1=pl.NUMERIC_DTYPES,
        )
        .defaults(null_strategy="ignore", null_param_1=None)
    )

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
    partition: Grouper,
    params: FrameType,
) -> FrameType:
    p = (
        ParamSchema(alpha=pl.Float64)
        .optional(
            min_periods=pl.Int64,
            adjust=pl.Boolean,
            null_strategy=pl.Categorical,
            null_param_1=pl.NUMERIC_DTYPES,
        )
        .defaults(
            min_periods=0, adjust=False, null_strategy="ignore", null_param_1=None
        )
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

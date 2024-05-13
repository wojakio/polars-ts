import polars as pl

from ..expr.mathx import diff_custom, ewm_custom, shift_custom

from ..sf_helper import impl_handle_null
from ..grouper import Grouper
from ..param_schema import ParamSchema

from ..types import FrameType


def impl_diff(
    df: FrameType,
    partition: Grouper,
    params: FrameType,
) -> FrameType:
    ps = ParamSchema(
        [
            ("diff", "n", pl.Int64, 1),
            ("diff", "method", pl.Categorical, "arithmetic"),
            ("null", "null_strategy", pl.Categorical, "ignore"),
            ("null", "null_param_1", pl.Float64, None),
        ]
    )

    df_fn, _params, result_cols = ps.apply("diff", df, params)

    grouper_cols = partition.apply(df_fn)
    numeric_cols = partition.numerics(df_fn, exclude=ps.names("*", invert=False))

    df_fn_result = df_fn.with_columns(
        diff_custom(
            pl.col(numeric_cols),
            "n",
            "method",
        ).over(grouper_cols)
    ).select(result_cols)

    _df_null, null_params, result_cols = ps.apply("null", df_fn_result, params)
    df_null_result = impl_handle_null(df_fn_result, partition, null_params)

    return df_null_result


def impl_cum_sum(df: FrameType, partition: Grouper) -> FrameType:
    p = ParamSchema([])
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df, exclude=p.names("*", invert=False))

    result = df.with_columns(pl.col(numeric_cols).cum_sum().over(grouper_cols))

    return result


def impl_shift(
    df: FrameType,
    partition: Grouper,
    params: FrameType,
) -> FrameType:
    ps = ParamSchema(
        [
            ("shift", "n", pl.Int64, 1),
            ("null", "null_strategy", pl.Categorical, "ignore"),
            ("null", "null_param_1", pl.Float64, None),
        ]
    )

    df, params, result_cols = ps.apply("shift", df, params)
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df, exclude=ps.names("*", invert=False))

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
    ps = ParamSchema(
        [
            ("ewm", "alpha", pl.Float64, 0.5),
            ("ewm", "min_periods", pl.Int64, 0),
            ("ewm", "adjust", pl.Boolean, False),
            ("ewm", "outlier_strategy", pl.String, None),
            ("ewm", "outlier_param_1", pl.Float64, 0.0),
            ("ewm", "outlier_param_2", pl.Float64, 100.0),
            ("null", "null_strategy", pl.Categorical, "ignore"),
            ("null", "null_param_1", pl.Float64, None),
        ]
    )

    df, _params, result_cols = ps.apply("ewm", df, params)
    grouper_cols = partition.apply(df)
    numeric_cols = partition.numerics(df, exclude=ps.names("*", invert=False))

    df_fn_result = df.with_columns(
        ewm_custom(
            pl.col(numeric_cols),
            "alpha",
            "min_periods",
            "adjust",
            "outlier_strategy",
            "outlier_param_1",
            "outlier_param_2",
        ).over(grouper_cols)
    ).select(result_cols)

    _df_null, null_params, result_cols = ps.apply("null", df_fn_result, params)
    df_null_result = impl_handle_null(df_fn_result, partition, null_params)

    return df_null_result

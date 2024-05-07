from typing import Any, Optional

import polars as pl
from polars.type_aliases import JoinStrategy

from ..types import cast_dtype, FrameType
from ..grouper import Grouper
from ..param_schema import ParamSchema
from ..expr.sf import handle_null_custom

RESERVED_COL_PREFIX = "##@_"
RESERVED_COL_REGEX = "^##@_.*$"
RESERVED_ALL_GRP = f"{RESERVED_COL_PREFIX}_GRP_ALL"
RESERVED_ROW_IDX = f"{RESERVED_COL_PREFIX}_INDEX"
RESERVED_DELIMITER = "##@"


def prepare_result(df: FrameType) -> FrameType:
    return df.select(pl.exclude(RESERVED_COL_REGEX))


def prepare_params(
    df: FrameType, params: Optional[FrameType], **kwargs: Any
) -> FrameType:
    if params is None:
        params = df.clear().select()

    params = params.with_columns(
        [
            cast_dtype(pl.lit(value), dtype).alias(name)
            for name, (value, dtype) in kwargs.items()
            if name not in params.columns
        ]
    )

    return params


def impl_handle_null(
    df: FrameType,
    partition: Grouper,
    params: FrameType,
) -> FrameType:
    ps = ParamSchema(
        [
            ("null", "null_strategy", pl.Categorical, "ignore"),
            ("null", "null_param_1", pl.Float64, None),
        ]
    )

    df, _params, result_cols = ps.apply("null", df, params)
    grouper_cols = partition.apply(df)
    # numeric_cols = partition.numerics(df, exclude=ps.names("*", invert=False))
    value_cols = partition.values(df, exclude=ps.names("*", invert=False))

    explode_cols = set(result_cols).difference(grouper_cols)

    result = (
        df.group_by(grouper_cols, maintain_order=True)
        .agg(
            pl.exclude(value_cols),
            handle_null_custom(pl.col(value_cols), "null_strategy", "null_param_1"),
        )
        .select(result_cols)
        # .with_columns(pl.col(explode_cols).list.lengths().name.prefix('len_'))
        .explode(*explode_cols)
    )

    return result


def impl_join(
    lhs: FrameType, rhs: FrameType, grouper: Grouper, how: JoinStrategy
) -> FrameType:
    grouper_cols = grouper.apply(lhs, rhs)
    df = lhs.join(rhs, on=grouper_cols, how=how)

    return df


def impl_unique(df: FrameType, grouper: Grouper) -> FrameType:
    grouper_cols = grouper.apply(df)
    df = df.unique(subset=grouper_cols, maintain_order=True)
    return df


def _add_unique_row_index(df: FrameType) -> FrameType:
    if RESERVED_ROW_IDX not in df.columns:
        df = df.with_row_index(name=RESERVED_ROW_IDX)

    return df


def column_name_unique_over(name: str, *df: FrameType) -> str:
    # orig_columns = df.columns
    unique_name = f"{RESERVED_COL_PREFIX}{name}"
    return unique_name


def impl_join_on_list_items(
    lhs: FrameType,
    rhs: FrameType,
    left_on: pl.Expr,
    right_on: pl.Expr,
    how: JoinStrategy,
    flatten: bool,
    then_unique: bool,
    then_sort: bool,
) -> FrameType:
    # left_on column has type: pl.List(some-dtype)
    # right_on column as type: some-dtype

    join_key_name = column_name_unique_over("join_key", lhs)

    def _make_join_key(xs: pl.Expr) -> pl.Expr:
        return xs.list.eval(pl.element().to_physical().hash()).list.sort().list.sum()

    result_expr = pl.exclude(left_on.meta.output_name(), right_on.meta.output_name())

    if flatten:
        result_expr = result_expr.explode()

        if then_unique:
            result_expr = result_expr.unique()

        if then_sort:
            result_expr = result_expr.sort()

    # build new rhs => construct unique join-keys present in lhs
    aggregated_rhs = (
        lhs.pipe(_add_unique_row_index)
        .select(RESERVED_ROW_IDX, left_on)
        .join(rhs, how="cross")
        .filter(right_on.is_in(left_on))
        .group_by(RESERVED_ROW_IDX)
        .agg(_make_join_key(left_on.first()).alias(join_key_name), result_expr)
        .drop(RESERVED_ROW_IDX)
        .sort(join_key_name)
        .unique(subset=join_key_name, maintain_order=True)
    )

    df = (
        lhs.with_columns(_make_join_key(left_on).alias(join_key_name)).join(
            aggregated_rhs, on=join_key_name, how=how
        )
        # .drop(join_key_name)
    )

    return df

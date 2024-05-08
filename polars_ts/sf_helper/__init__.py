from typing import Any, List, Optional

import polars as pl
from polars.type_aliases import JoinStrategy

from ..types import cast_dtype, FrameType
from ..grouper import Grouper
from ..param_schema import ParamSchema

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


def _handle_null_helper(df: FrameType, value_cols: List[str]) -> FrameType:
    first_row = df.lazy().head(1).collect()
    null_strategy = first_row.item(0, "null_strategy")

    match null_strategy:
        case "sentinel":
            sentinel_value = first_row.item(0, "null_param_1")
            result = df.with_columns(pl.col(value_cols).fill_null(sentinel_value))
        case "forward":
            fill_limit = int(first_row.item(0, "null_param_1"))
            result = df.with_columns(pl.col(value_cols).forward_fill(fill_limit))
        case "backward":
            fill_limit = int(first_row.item(0, "null_param_1"))
            result = df.with_columns(pl.col(value_cols).backward_fill(fill_limit))
        case "interpolate_linear":
            result = df.with_columns(pl.col(value_cols).interpolate(method="linear"))
        case "interpolate_nearest":
            result = df.with_columns(pl.col(value_cols).interpolate(method="nearest"))
        case "min":
            result = df.with_columns(pl.col(value_cols).fill_null(strategy="min"))
        case "max":
            result = df.with_columns(pl.col(value_cols).fill_null(strategy="max"))
        case "mean":
            result = df.with_columns(pl.col(value_cols).fill_null(strategy="mean"))
        case "ignore":
            result = df
        case "trim_start_n":
            n = int(first_row.item(0, "null_param_1"))
            result = df.slice(n)
        case "trim_end_n":
            n = int(first_row.item(0, "null_param_1"))
            result = df.reverse().slice(n).reverse()
        case "drop_if_all":
            result = df.filter(pl.all_horizontal(pl.col(value_cols).is_not_null()))
        case "drop_if_any":
            result = df.filter(pl.any_horizontal(pl.col(value_cols).is_not_null()))
        case _:
            raise ValueError(f"Unknown null_strategy {null_strategy}")

    return result


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

    result = (
        df.group_by(grouper_cols, maintain_order=True)
        .map_groups(lambda df: _handle_null_helper(df, value_cols), df.schema)  # type: ignore[call-arg]
        .select(result_cols)
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

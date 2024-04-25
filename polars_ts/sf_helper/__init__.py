import polars as pl
from polars.type_aliases import JoinStrategy

from ..types import FrameType, NullStrategyType, SentinelNumeric
from ..grouper import Grouper

RESERVED_COL_PREFIX = "##@_"
RESERVED_COL_REGEX = "^##@_.*$"
RESERVED_ALL_GRP = f"{RESERVED_COL_PREFIX}_GRP_ALL"
RESERVED_ROW_IDX = f"{RESERVED_COL_PREFIX}_INDEX"
RESERVED_DELIMITER = "##@"


def prepare_result(df: FrameType) -> FrameType:
    return df.select(pl.exclude(RESERVED_COL_REGEX))


def impl_fill_null(
    df: FrameType,
    null_strategy: NullStrategyType,
    null_sentinel: SentinelNumeric,
    partition: Grouper,
) -> FrameType:
    grouper_cols = partition.apply(df)

    if null_strategy == "sentinel":
        result = df.with_columns(pl.col(pl.NUMERIC_DTYPES).fill_null(null_sentinel))

    elif null_strategy == "forward":
        result = df.with_columns(pl.exclude("time").forward_fill().over(grouper_cols))

    elif null_strategy == "backward":
        result = df.with_columns(pl.exclude("time").backward_fill().over(grouper_cols))

    elif null_strategy == "interpolate_linear":
        result = df.with_columns(
            pl.exclude("time").interpolate(method="linear").over(grouper_cols)
        )

    elif null_strategy == "interpolate_nearest":
        result = df.with_columns(
            pl.exclude("time").interpolate(method="nearest").over(grouper_cols)
        )

    elif null_strategy == "min":
        result = df.with_columns(pl.exclude("time").min().over(grouper_cols))

    elif null_strategy == "max":
        result = df.with_columns(pl.exclude("time").max().over(grouper_cols))

    elif null_strategy == "mean":
        result = df.with_columns(pl.exclude("time").mean().over(grouper_cols))

    elif null_strategy == "ignore":
        result = df

    elif null_strategy == "drop":
        cols = set(df.columns).difference(["time"] + grouper_cols)
        result = df.drop_nulls(subset=cols)

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

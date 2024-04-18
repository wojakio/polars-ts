from typing import Union
import polars as pl
from polars.type_aliases import JoinStrategy

from ..types import FrameType, NullStrategyType
from ..grouper import Grouper

RESERVED_COL_PREFIX = "##@_"
RESERVED_COL_REGEX = "^##@_.*$"
RESERVED_ALL_GRP = f"{RESERVED_COL_PREFIX}_GRP_ALL"
RESERVED_ROW_IDX = f"{RESERVED_COL_PREFIX}_INDEX"


def prepare_result(df: FrameType) -> FrameType:
    return df.select(pl.exclude(RESERVED_COL_REGEX))


def impl_apply_null_strategy(
    df: FrameType,
    null_strategy: NullStrategyType,
    null_sentinel_numeric: Union[float, int],
    partition: Grouper,
) -> FrameType:
    grouper_cols = partition.apply(df)

    if null_strategy == "sentinel_numeric":
        result = df.with_columns(
            pl.col(pl.NUMERIC_DTYPES).fill_null(null_sentinel_numeric)
        )

    elif null_strategy == "forward":
        result = df.with_columns(pl.exclude("time").forward_fill().over(grouper_cols))

    elif null_strategy == "backward":
        result = df.with_columns(pl.exclude("time").backward_fill().over(grouper_cols))

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

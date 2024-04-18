from typing import Literal

import polars as pl

from ..grouper import Grouper

from ..types import FrameType, NullBehaviorType


def impl_diff(
    df: FrameType,
    k: int,
    method: Literal["arithmetic", "fractional", "geometric"],
    partition: Grouper,
    null_behavior: NullBehaviorType,
) -> FrameType:
    grouper_cols = partition.apply(df)
    value_cols = partition.numeric(df)

    result = df.with_columns(
        pl.col(value_cols).diff(k, null_behavior="ignore").over(grouper_cols)
    )

    if null_behavior == "drop":
        result = result.drop_nulls(subset=value_cols)

    return result

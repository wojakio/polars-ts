from typing import Union, List

import polars as pl

from ..sf_helper import impl_categories
from ..grouper import Grouper

from ..types import (
    IntervalType,
    FillStrategyType,
    RetainValuesType,
    FrameType,
)


def impl_align_to_time(
    df: FrameType,
    time_axis: pl.Series,
    partition: Grouper,
    closed: IntervalType,
    fill_strategy: FillStrategyType,
    fill_sentinel: Union[float, int],
) -> FrameType:
    resampled = impl_resample_categories(df, time_axis, partition, closed)
    realigned = impl_align_values(
        df,
        resampled,
        partition,
        retain_values="lhs",
        fill_strategy=fill_strategy,
        fill_sentinel=fill_sentinel,
    )

    return realigned


def impl_resample_categories(
    df: FrameType,
    time_axis: pl.Series,
    partition: Grouper,
    closed: IntervalType,
) -> FrameType:
    grouper_cols = partition.apply(df)

    grouper_cols = partition.apply(df)
    categories = df.select(grouper_cols).unique(maintain_order=True)
    new_index = df.select(time=time_axis).join(categories, how="cross")

    if closed != "none":
        start_ends = df.select(
            *grouper_cols,
            start=pl.max_horizontal(
                pl.col("time").min().over(grouper_cols), time_axis.min()
            ),
            end=pl.min_horizontal(
                pl.col("time").max().over(grouper_cols), time_axis.max()
            ),
        ).unique(keep="first", maintain_order=True)

        new_index = new_index.join(start_ends, on=grouper_cols)

    if not (closed == "left" or closed == "none"):
        new_index = new_index.filter(pl.col("time") <= pl.col("end"))

    if not (closed == "right" or closed == "none"):
        new_index = new_index.filter(pl.col("time") >= pl.col("start"))

    new_index = new_index.select(pl.exclude("start", "end")).sort("time", *grouper_cols)

    return new_index


def impl_align_values(
    lhs: FrameType,
    rhs: FrameType,
    partition: Grouper,
    retain_values: RetainValuesType,
    fill_strategy: FillStrategyType,
    fill_sentinel: Union[float, int],
) -> FrameType:
    if retain_values == "lhs":
        # strip away rhs values
        rhs = impl_categories(rhs, include_time=True)

    if retain_values == "rhs":
        # strip away lhs values
        lhs = impl_categories(lhs, include_time=True)

    assert isinstance(lhs, type(rhs)) and (
        isinstance(lhs, pl.LazyFrame) or isinstance(lhs, pl.DataFrame)
    )

    grouper_cols: List[str] = partition.by_common().apply(lhs, rhs)

    result = (
        lhs.join(rhs, on=["time", *grouper_cols], how="outer_coalesce")
        .filter(pl.col("time").is_not_null())
        .sort("time", *grouper_cols)
        .unique(keep="first", maintain_order=True)
    )

    if fill_strategy == "sentinel":
        result = result.with_columns(pl.col(pl.NUMERIC_DTYPES).fill_null(fill_sentinel))

    elif fill_strategy == "forward":
        result = result.with_columns(
            pl.exclude("time").forward_fill().over(grouper_cols)
        )

    elif fill_strategy == "backward":
        result = result.with_columns(
            pl.exclude("time").backward_fill().over(grouper_cols)
        )

    elif fill_strategy == "none":
        pass

    else:
        raise ValueError(f"Unexpected fill strategy: {fill_strategy}")

    rhs_grid = impl_categories(rhs, include_time=True)
    result = result.join(rhs_grid, on=rhs_grid.columns, how="inner")

    return result

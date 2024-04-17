from typing import Union

import polars as pl

from ..sf_helper import impl_auto_partition, impl_categories, impl_common_category_names

from ..types import (
    PartitionType,
    IntervalType,
    FillStrategyType,
    RetainValuesType,
    FrameType,
)


def impl_align_to_time(
    df: FrameType,
    time_axis: pl.Series,
    partition: PartitionType,
    closed: IntervalType,
    fill_strategy: FillStrategyType,
    fill_sentinel: Union[float, int],
) -> FrameType:
    partition = impl_auto_partition(df, partition)
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
    partition: PartitionType,
    closed: IntervalType,
) -> FrameType:
    partition = impl_auto_partition(df, partition)

    categories = df.select(partition).unique()
    new_index = df.select(time=time_axis).join(categories, how="cross")

    if closed != "none":
        start_ends = df.select(
            *partition,
            start=pl.max_horizontal(
                pl.col("time").min().over(partition), time_axis.min()
            ),
            end=pl.min_horizontal(
                pl.col("time").max().over(partition), time_axis.max()
            ),
        ).unique(keep="first", maintain_order=True)

        new_index = new_index.join(start_ends, on=partition)

    if not (closed == "left" or closed == "none"):
        new_index = new_index.filter(pl.col("time") <= pl.col("end"))

    if not (closed == "right" or closed == "none"):
        new_index = new_index.filter(pl.col("time") >= pl.col("start"))

    new_index = new_index.select(pl.exclude("start", "end")).sort("time", *partition)

    return new_index


def impl_align_values(
    lhs: FrameType,
    rhs: FrameType,
    partition: PartitionType,
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

    if partition is None:
        partition = impl_common_category_names(lhs, rhs)

    assert isinstance(lhs, type(rhs)) and (
        isinstance(lhs, pl.LazyFrame) or isinstance(lhs, pl.DataFrame)
    )

    result = (
        lhs.join(rhs, on=["time", *partition], how="outer_coalesce")
        .filter(pl.col("time").is_not_null())
        .sort("time", *partition)
        .unique(keep="first", maintain_order=True)
    )

    if fill_strategy == "sentinel":
        result = result.with_columns(pl.col(pl.NUMERIC_DTYPES).fill_null(fill_sentinel))

    elif fill_strategy == "forward":
        result = result.with_columns(pl.exclude("time").forward_fill().over(partition))

    elif fill_strategy == "backward":
        result = result.with_columns(pl.exclude("time").backward_fill().over(partition))

    elif fill_strategy == "none":
        pass

    else:
        raise ValueError(f"Unexpected fill strategy: {fill_strategy}")

    rhs_grid = impl_categories(rhs, include_time=True)
    result = result.join(rhs_grid, on=rhs_grid.columns, how="inner")

    return result

import polars as pl

from ..sf_helper import impl_fill_null
from ..grouper import Grouper

from ..types import (
    IntervalType,
    NullStrategyType,
    RetainValuesType,
    FrameType,
    SentinelNumeric,
)


def impl_align_to_time(
    df: FrameType,
    time_axis: pl.Series,
    partition: Grouper,
    closed: IntervalType,
    null_strategy: NullStrategyType,
    null_sentinel: SentinelNumeric,
) -> FrameType:
    resampled = impl_resample_categories(df, time_axis, partition, closed)
    realigned = impl_align_values(
        df,
        resampled,
        partition,
        retain_values="lhs",
        null_strategy=null_strategy,
        null_sentinel=null_sentinel,
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
    null_strategy: NullStrategyType,
    null_sentinel: SentinelNumeric,
) -> FrameType:
    rhs_cat_cols = Grouper.categories(rhs, include_time=True)
    rhs_grid = rhs.select(rhs_cat_cols)

    if retain_values == "lhs":
        # strip away rhs values
        rhs = rhs_grid

    if retain_values == "rhs":
        # strip away lhs values
        lhs_cat_cols = Grouper.categories(lhs, include_time=True)
        lhs = lhs.select(lhs_cat_cols)

    common_cols = Grouper.by_common_including_time().apply(lhs, rhs)

    result = (
        lhs.join(rhs, on=common_cols, how="outer_coalesce")
        .filter(pl.col("time").is_not_null())
        .sort(*common_cols)
        .unique(keep="first", maintain_order=True)
    )

    result = impl_fill_null(result, null_strategy, null_sentinel, partition)

    result = result.join(rhs_grid, on=rhs_cat_cols, how="inner")

    return result

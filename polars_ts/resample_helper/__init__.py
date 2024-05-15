from typing import Callable

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.interpolate import CubicSpline

from ..sf_helper import RESERVED_COL_REGEX
from ..sf_helper import impl_handle_null
from ..grouper import Grouper

from ..types import (
    IntervalType,
    RetainValuesType,
    FrameType,
    SentinelNumeric,
)


def impl_fit_spline(df: FrameType, interpolation: str, output_name: str) -> FrameType:
    fn_data = (
        df.lazy()
        .collect()
        .select(pl.exclude(RESERVED_COL_REGEX).cast(pl.Float64, strict=False))
        .melt(id_vars="x", variable_name="name", value_name="y")
        .with_columns(pl.col("name").cast(pl.Categorical))
    )

    fns = {
        "cubic_clamped": (CubicSpline, {"bc_type": "clamped"}),
        "cubic_natural": (CubicSpline, {"bc_type": "natural"}),
    }

    def _fitted_fn(fargs: pl.Series) -> Callable[..., npt.ArrayLike]:
        xs = fargs.struct.field("x")
        ys = fargs.struct.field("y")

        fn, fn_args = fns[interpolation]
        raw_fn: Callable[[npt.ArrayLike], npt.ArrayLike] = fn(xs, ys, **fn_args)

        def clipped_fn(x: npt.ArrayLike) -> npt.ArrayLike:
            a_min: npt.ArrayLike = xs.min()  # type: ignore[assignment]
            a_max: npt.ArrayLike = xs.max()  # type: ignore[assignment]
            clipped_x: npt.ArrayLike = np.clip(x, a_min, a_max)
            return raw_fn(clipped_x)

        return clipped_fn

    result = (
        fn_data.group_by("name")
        .agg(fn=pl.struct("x", "y").map_batches(_fitted_fn))
        .explode("fn")
        .rename(
            {
                "name": f"{output_name}",
                "fn": f"{output_name}_fn",
            }
        )
        .lazy()
    )

    typed_result = result if isinstance(df, pl.LazyFrame) else result.collect()
    return typed_result


def impl_align_to_time(
    df: FrameType,
    time_axis: pl.Series,
    partition: Grouper,
    closed: IntervalType,
    null_strategy: str,
    null_param_1: SentinelNumeric,
) -> FrameType:
    resampled = impl_resample_categories(df, time_axis, partition, closed)
    realigned = impl_align_values(
        df,
        resampled,
        partition,
        retain_values="lhs",
        null_strategy=null_strategy,
        null_param_1=null_param_1,
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
    null_strategy: str,
    null_param_1: SentinelNumeric,
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

    null_params = lhs.select(
        null_strategy=pl.lit(null_strategy).cast(pl.Categorical),
        null_param_1=pl.lit(null_param_1).cast(pl.Float64),
    )

    result = impl_handle_null(result, partition, null_params)

    result = result.join(rhs_grid, on=rhs_cat_cols, how="inner")

    return result

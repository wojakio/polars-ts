from datetime import datetime
from typing import Optional

import polars as pl


from ..types import (
    FrameType,
)


def impl_range(
    df: FrameType,
    t0: datetime,
    t1: Optional[datetime] = None,
    offset: str = "0d",
    interval: str = "1d",
) -> FrameType:
    result = df.with_columns(
        time=(
            pl.lit([t0, t1])
            .list.concat(pl.lit(t0).dt.offset_by(offset))
            .list.drop_nulls()
            .list.unique()
            .list.sort()
        )
    )

    return impl_expand(result, interval)


def impl_expand(df: FrameType, interval: str) -> FrameType:
    # pre: every row has a time: list[date] column
    # xdt doesnt support date_ranges?
    # fn = xdt.date_range if interval.endswith('bd') else pl.date_ranges
    fn = pl.date_ranges

    result = df.with_columns(
        fn(pl.col("time").list.min(), pl.col("time").list.max(), interval)
    ).explode("time")

    return result

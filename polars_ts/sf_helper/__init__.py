import polars as pl

from ..types import FrameType

RESERVED_COL_PREFIX = "##@_"
RESERVED_COL_REGEX = "^##@_.*$"
RESERVED_ALL_GRP = f"{RESERVED_COL_PREFIX}_GRP_ALL"
RESERVED_ROW_IDX = f"{RESERVED_COL_PREFIX}_INDEX"


def prepare_result(df: FrameType) -> FrameType:
    return df.select(pl.exclude(RESERVED_COL_REGEX))


def impl_categories(df: FrameType, include_time: bool = False) -> FrameType:
    if include_time:
        return df.select("time", pl.col(pl.Categorical, pl.Enum))

    return df.select(pl.col(pl.Categorical, pl.Enum))

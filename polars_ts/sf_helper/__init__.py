from typing import List

import polars as pl

from ..types import FrameType, PartitionType

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


def impl_common_category_names(lhs: FrameType, rhs: FrameType) -> List[str]:
    lhs_categories = set(impl_categories(lhs).columns)
    rhs_categories = impl_categories(rhs).columns
    return sorted(lhs_categories.intersection(rhs_categories))


def impl_auto_partition(
    df: FrameType,
    partition: PartitionType,
) -> List[str]:
    cols = set(impl_categories(df).columns).difference(["time"])

    if partition is None or isinstance(partition, list):
        return sorted(cols)

    if len(set(partition.keys()).intersection(["by", "but"])) > 1:
        raise ValueError("Must specify 'by' xor 'but'")

    if "by" in partition:
        by = partition["by"]
        cols = cols.intersection(by)

    if "but" in partition:
        but = partition["but"]
        cols = cols.difference(but)

    if len(cols) == 0:
        raise ValueError(f"Invalid partition spec. {partition}")

    return sorted(cols)

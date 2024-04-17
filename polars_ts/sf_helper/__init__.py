from typing import List, Mapping, Literal, Union

import polars as pl

RESERVED_COL_PREFIX = "##@_"
RESERVED_COL_REGEX = "^##@_.*$"
RESERVED_ALL_GRP = f"{RESERVED_COL_PREFIX}_GRP_ALL"
RESERVED_ROW_IDX = f"{RESERVED_COL_PREFIX}_INDEX"

def prepare_result(df: Union[pl.LazyFrame, pl.DataFrame]) -> pl.LazyFrame:
    return df.select(pl.exclude(RESERVED_COL_REGEX))

def impl_categories(df: Union[pl.LazyFrame, pl.DataFrame], include_time: bool = False) -> pl.LazyFrame:
    if include_time:
        return df.select("time", pl.col(pl.Categorical, pl.Enum))

    return df.select(pl.col(pl.Categorical, pl.Enum))

def impl_common_category_names(lhs: pl.LazyFrame, rhs: pl.LazyFrame) -> List[str]:
    lhs_categories = set(impl_categories(lhs).columns)
    rhs_categories = impl_categories(rhs).columns
    return sorted(lhs_categories.intersection(rhs_categories))

def impl_auto_partition(
    df: pl.LazyFrame,
    partition: Union[Mapping[Literal["by", "but"], List[str]] | None],
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

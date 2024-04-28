from typing import Literal, TypeVar, Union
import polars as pl

FrameType = TypeVar("FrameType", pl.LazyFrame, pl.DataFrame)

RetainValuesType = Literal["lhs", "rhs", "both"]
IntervalType = Literal["none", "left", "right", "both"]

CorrelationType = Literal[
    "additive", "multiplicative", "shift", "exponent", "average", "none"
]

NullStrategyType = Literal[
    "ignore",
    # drop strategies - alters length of series
    "drop",
    # "drop_at_start",
    # "drop_at_end",
    # "drop_n",
    # fill strategies
    "sentinel",
    "forward",
    "backward",
    "interpolate_linear",
    "interpolate_nearest",
    "min",
    "max",
    "mean",
]

SentinelNumeric = Union[float, int]

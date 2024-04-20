from typing import Literal, TypeVar, Union
import polars as pl

FrameType = TypeVar("FrameType", pl.LazyFrame, pl.DataFrame)

RetainValuesType = Literal["lhs", "rhs", "both"]
IntervalType = Literal["none", "left", "right", "both"]

CorrelationType = Literal[
    "additive", "multiplicative", "shift", "exponent", "average", "none"
]

NullStrategyType = Literal[
    "drop",
    "ignore",
    "sentinel",
    "forward",
    "backward",
    "interpolate_linear",
    "interpolate_nearest",
]

SentinelNumeric = Union[float, int]

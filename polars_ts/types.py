from typing import Literal, TypeVar
import polars as pl

FrameType = TypeVar("FrameType", pl.LazyFrame, pl.DataFrame)

RetainValuesType = Literal["lhs", "rhs", "both"]
FillStrategyType = Literal["none", "sentinel", "forward", "backward"]
IntervalType = Literal["none", "left", "right", "both"]

CorrelationType = Literal[
    "additive", "multiplicative", "shift", "exponent", "average", "none"
]

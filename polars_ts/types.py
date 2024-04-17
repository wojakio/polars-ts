from typing import Literal, Optional, Mapping, List, Union
import polars as pl

FrameType = Union[pl.LazyFrame, pl.DataFrame]

RetainValuesType = Literal["lhs", "rhs", "both"]
FillStrategyType = Literal["none", "sentinel", "forward", "backward"]
IntervalType = Literal["none", "left", "right", "both"]
PartitionType = Optional[Mapping[Literal["by", "but"], List[str]]]

CorrelationType = Literal[
    "additive", "multiplicative", "shift", "exponent", "average", "none"
]

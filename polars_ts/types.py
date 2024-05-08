from typing import Literal, TypeVar, Union
import polars as pl


def cast_dtype(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    if dtype is None:
        return expr

    if dtype == pl.NUMERIC_DTYPES:
        return expr

    if dtype == pl.Categorical:
        expr = expr.cast(pl.String)

    return expr.cast(dtype)


FrameType = TypeVar("FrameType", pl.LazyFrame, pl.DataFrame)

RetainValuesType = Literal["lhs", "rhs", "both"]
IntervalType = Literal["none", "left", "right", "both"]

CorrelationType = Literal[
    "additive", "multiplicative", "shift", "exponent", "average", "none"
]

SentinelNumeric = Union[float, int]

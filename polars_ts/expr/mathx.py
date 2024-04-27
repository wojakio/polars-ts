from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

from ..utils import parse_into_expr


def ewm_custom(
    values_expr: IntoExpr,
    alpha: IntoExpr,
    min_periods: IntoExpr,
    adjust: IntoExpr,
) -> pl.Expr:
    values_expr = parse_into_expr(values_expr)
    alpha = parse_into_expr(alpha).cast(pl.Float64)
    min_periods = parse_into_expr(min_periods).cast(pl.UInt64)
    adjust = parse_into_expr(adjust).cast(pl.Boolean)

    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent.parent,
        function_name="pl_ewm_custom",
        args=[values_expr, alpha, min_periods, adjust],
        is_elementwise=False,
    )

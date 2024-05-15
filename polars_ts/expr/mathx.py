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
    outlier_strategy: IntoExpr,
    outlier_param_1: IntoExpr,
    outlier_param_2: IntoExpr,
) -> pl.Expr:
    values_expr = parse_into_expr(values_expr)
    alpha = parse_into_expr(alpha).cast(pl.Float64)
    min_periods = parse_into_expr(min_periods).cast(pl.UInt64)
    adjust = parse_into_expr(adjust).cast(pl.Boolean)
    outlier_strategy = parse_into_expr(outlier_strategy).cast(pl.String)
    outlier_param_1 = parse_into_expr(outlier_param_1).cast(pl.Float64)
    outlier_param_2 = parse_into_expr(outlier_param_2).cast(pl.Float64)

    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent.parent,
        function_name="pl_ewm_custom",
        args=[
            values_expr,
            alpha,
            min_periods,
            adjust,
            outlier_strategy,
            outlier_param_1,
            outlier_param_2,
        ],
        is_elementwise=False,
    )


def shift_custom(
    values_expr: IntoExpr,
    n: IntoExpr,
) -> pl.Expr:
    values_expr = parse_into_expr(values_expr)
    n = parse_into_expr(n).cast(pl.Int64)

    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent.parent,
        function_name="pl_shift_custom",
        args=[values_expr, n],
        is_elementwise=False,
    )


def diff_custom(
    values_expr: IntoExpr,
    n: IntoExpr,
    method: IntoExpr,
) -> pl.Expr:
    values_expr = parse_into_expr(values_expr)
    n = parse_into_expr(n).cast(pl.Int64)
    method = parse_into_expr(method).cast(pl.String)

    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent.parent,
        function_name="pl_diff_custom",
        args=[values_expr, n, method],
        is_elementwise=False,
    )

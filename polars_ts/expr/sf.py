from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

from ..utils import parse_into_expr


def handle_null_custom(
    values_expr: IntoExpr,
    null_strategy: IntoExpr,
    null_param_1: IntoExpr,
) -> pl.Expr:
    values_expr = parse_into_expr(values_expr)
    null_strategy = parse_into_expr(null_strategy).cast(pl.String)
    null_param_1 = parse_into_expr(null_param_1)

    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent.parent,
        function_name="pl_handle_null_custom",
        args=[values_expr, null_strategy, null_param_1],
        is_elementwise=False,
    )

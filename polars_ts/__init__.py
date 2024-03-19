from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


def pig_latinnify(expr: IntoExpr) -> pl.Expr:
    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="pig_latinnify",
        args=expr,
        is_elementwise=True,
    )

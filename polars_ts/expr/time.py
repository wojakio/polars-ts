from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

from ..utils import parse_into_expr


def datetime_ranges_custom(
    start_date: IntoExpr,
    end_date: IntoExpr,
    skip_dates: IntoExpr,
    iso_weekends: IntoExpr,
) -> pl.Expr:
    start_date = parse_into_expr(start_date)
    end_date = parse_into_expr(end_date)
    skip_dates = parse_into_expr(skip_dates)
    iso_weekends = parse_into_expr(iso_weekends)

    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent.parent,
        function_name="pl_datetime_ranges_custom",
        args=[start_date, end_date, skip_dates, iso_weekends],
        is_elementwise=False,
    )

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


def wyhash(expr: IntoExpr) -> pl.Expr:
    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent.parent,
        function_name="pl_wyhash",
        args=expr,
        is_elementwise=True,
    )


def random_uniform(
    lower: Union[pl.Expr, float] = 0.0,
    upper: Union[pl.Expr, float] = 1.0,
    seed: Union[pl.Expr, int] = 42,
) -> pl.Expr:
    lo = pl.lit(lower, pl.Float64) if isinstance(lower, float) else lower
    up = pl.lit(upper, pl.Float64) if isinstance(upper, float) else upper
    seed = pl.lit(seed, pl.UInt64) if isinstance(seed, int) else seed

    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent.parent,
        function_name="pl_random_uniform",
        args=[pl.len(), lo, up, seed],
        is_elementwise=False,
    )


def random_normal(
    mu: Union[pl.Expr, float] = 0.0,
    sigma: Union[pl.Expr, float] = 1.0,
    seed: Union[pl.Expr, int] = 42,
) -> pl.Expr:
    mean = pl.lit(mu, pl.Float64) if isinstance(mu, float) else mu
    var = pl.lit(sigma, pl.Float64) if isinstance(sigma, float) else sigma
    seed = pl.lit(seed, pl.UInt64) if isinstance(seed, int) else seed

    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent.parent,
        function_name="pl_random_normal",
        args=[pl.len(), mean, var, seed],
        is_elementwise=False,
    )

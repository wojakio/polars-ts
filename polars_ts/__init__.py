from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from .display import *
from .sf import *
from .tsf import *
from .io import *
from .time import *
from .convert import *
from .dummy import *
from .mathx import *
from .resample import *

from .calendar import *
from .futures import *
from .dummymkt import *

from .grouper import Grouper  # noqa

pl.enable_string_cache()


if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


def pig_latinnify(expr: IntoExpr) -> pl.Expr:
    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="pig_latinnify",
        args=expr,
        is_elementwise=True,
    )


def template_1(expr: IntoExpr, *, seed: int) -> pl.Expr:
    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="pl_template_1",
        args=expr,
        kwargs={"seed": seed},
        is_elementwise=False,
    )

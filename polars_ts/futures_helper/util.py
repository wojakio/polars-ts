from typing import Dict, Union

import polars as pl
from polars.type_aliases import IntoExpr

from ..utils import parse_into_expr


def month_to_imm_dict(invert: bool = False) -> Dict[Union[str, int], Union[str, int]]:
    result_cols = ["month_idx", "imm_code"]
    df = (
        pl.from_dict({"imm_code": "FGHJKMNQUVXZ"})
        .select(
            pl.col("imm_code").str.extract_all("[F-Z]").explode().cast(pl.Categorical)
        )
        .with_row_index("month_idx", offset=1)
        .select(reversed(result_cols) if invert else result_cols)
    )

    return dict(df.iter_rows())


def make_generic_contract(col_or_date: IntoExpr) -> pl.Expr:
    mo2imm_dict = month_to_imm_dict()
    col_expr = parse_into_expr(col_or_date, dtype=pl.Date)

    return pl.struct(
        tenor=col_expr.dt.month().replace(mo2imm_dict).cast(pl.Categorical),
        year=col_expr.dt.year(),
    )

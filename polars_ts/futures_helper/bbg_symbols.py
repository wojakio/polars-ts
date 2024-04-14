import polars as pl
from polars.type_aliases import IntoExpr

from ..utils import parse_into_expr


def to_bbg_instrument(
    instrument_id: IntoExpr, symbol: IntoExpr, yellow_key: IntoExpr = pl.lit("")
) -> pl.Expr:
    instrument_id_expr = parse_into_expr(instrument_id)
    symbol_expr = parse_into_expr(symbol).cast(pl.String)
    yellow_key_expr = parse_into_expr(yellow_key).cast(pl.String)

    return pl.concat_str(
        [
            symbol_expr,
            instrument_id_expr.struct.field("tenor").cast(pl.String),
            instrument_id_expr.struct.field("year").cast(pl.String).str.tail(2),
            (
                pl.when(yellow_key_expr.str.len_chars() == 0)
                .then(yellow_key_expr)
                .otherwise(pl.concat_str(pl.lit(" "), yellow_key_expr))
            ),
        ]
    ).cast(pl.Categorical)

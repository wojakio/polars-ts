from typing import Optional

import polars as pl
from polars.type_aliases import IntoExpr

from .sf import SeriesFrame
from .futures_helper.util import month_to_imm_dict
from .utils import parse_into_expr


__NAMESPACE = "calendar"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class CalendarFrame(SeriesFrame):
    def __init__(self, df: pl.LazyFrame):
        super().__init__(df)

    def date_to_imm_contract(
        self,
        col_name: IntoExpr,
        prefix: IntoExpr = pl.lit(""),
        suffix: IntoExpr = pl.lit(""),
        out: Optional[str] = None,
    ) -> pl.LazyFrame:

        col_expr = parse_into_expr(col_name)
        prefix = parse_into_expr(prefix)
        suffix = parse_into_expr(suffix)
        out = col_expr.meta.output_name() if out is None else out

        imm_dict = month_to_imm_dict()
        self._df = self._df.with_columns(
            pl.when(col_expr.is_null())
            .then(pl.lit(None).cast(pl.String))
            .otherwise(
                pl.concat_str([
                    prefix,
                    pl.struct(
                        col_expr.dt.month().alias("month"),
                        col_expr.dt.strftime("%y").alias("year"),
                    )
                    .map_elements(
                        lambda se: f"""{imm_dict.get(se['month'], "@@")}{se['year']}""",
                        return_dtype=pl.String,
                    ),
                    suffix
                ])
            )
            .alias(out)
        )

        return self.result_df

    def imm_contract_to_date(self, col_name: str, default_day: int = 1) -> pl.LazyFrame:
        imm_dict = month_to_imm_dict(invert=True)
        self._df = self._df.with_columns(
            pl.col(col_name)
            .str.extract_groups(".*(?<imm>[F-Z])(?<year>[0-9]{2})")
            .map_elements(
                lambda se: (
                    f"{str(default_day).rjust(2, '0')}-{str(imm_dict[se['imm']]).rjust(2, '0')}-{se['year']}"
                ),
                return_dtype=pl.String,
            )
            .str.to_date("%d-%m-%y")
        )

        return self.result_df

    def int_to_date(self, col_name: str) -> pl.LazyFrame:
        self._df = self._df.with_columns(
            pl.col(col_name).cast(pl.String).str.strptime(pl.Date, "%Y%m%d")
        )

        return self.result_df

    def date_to_int(self, col_name: str, out: Optional[str] = None) -> pl.LazyFrame:
        out = col_name if out is None else out

        self._df = self._df.with_columns(
            (
                (pl.col(col_name).dt.year() * 10000)
                + (pl.col(col_name).dt.month().cast(pl.UInt16) * 100)
                + (pl.col(col_name).dt.day())
            )
            .alias(out)
            .cast(pl.UInt64)
        )

        return self.result_df

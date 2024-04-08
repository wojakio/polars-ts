from typing import Optional, Union

import polars as pl

from .sf import SeriesFrame
from .futures_helper.util import month_to_imm


__NAMESPACE = "calendar"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class CalendarFrame(SeriesFrame):
    def __init__(self, df: pl.LazyFrame):
        super().__init__(df)

    def date_to_imm_contract(
        self,
        col_name: str,
        contract_code: Union[pl.Expr, str] = "",
        out: Optional[str] = None,
    ) -> pl.Expr:
        imm_dict = month_to_imm(as_dict=True)
        out = col_name if out is None else out
        contract_code = (
            pl.lit(contract_code) if isinstance(contract_code, str) else contract_code
        )

        self._df = self._df.with_columns(
            pl.concat_str(
                [
                    contract_code,
                    pl.struct(
                        pl.col(col_name).dt.month().alias("month"),
                        pl.col(col_name).dt.strftime("%y").alias("year"),
                    ).map_elements(
                        lambda se: f"{imm_dict[int(se['month'])]}{se['year']}",
                        return_dtype=pl.String,
                    ),
                ]
            ).alias(out)
        )

        return self.result_df

    def imm_contract_to_date(self, col_name: str, default_day: int = 1) -> pl.Expr:
        imm_dict = month_to_imm(as_dict=True, invert=True)
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

    def int_to_date(self, col_name: str) -> pl.Expr:
        self._df = self._df.with_columns(
            pl.col(col_name).cast(pl.String).str.strptime(pl.Date, "%Y%m%d")
        )

        return self.result_df

    def date_to_int(self, col_name: str, out: Optional[str] = None) -> pl.Expr:
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

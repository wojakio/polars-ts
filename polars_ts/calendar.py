from typing import Optional, Generic

import polars as pl
from polars.type_aliases import IntoExpr

from .sf import SeriesFrame
from .types import FrameType
from .sf_helper import prepare_result
from .calendar_helper import (
    impl_date_to_imm_contract,
    impl_date_to_int,
    impl_imm_contract_to_date,
    impl_int_to_date,
)

from .utils import parse_into_expr

__NAMESPACE = "calendar"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class CalendarFrame(SeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType):
        super().__init__(df)

    def date_to_imm_contract(
        self,
        col_name: IntoExpr,
        prefix: IntoExpr = pl.lit(""),
        suffix: IntoExpr = pl.lit(""),
        out: Optional[str] = None,
    ) -> FrameType:
        col_name = parse_into_expr(col_name)
        prefix = parse_into_expr(prefix)
        suffix = parse_into_expr(suffix)
        out = col_name.meta.output_name() if out is None else out

        df = impl_date_to_imm_contract(self._df, col_name, prefix, suffix, out)

        return prepare_result(df)

    def imm_contract_to_date(self, col_name: str, default_day: int = 1) -> FrameType:
        df = impl_imm_contract_to_date(self._df, col_name, default_day)
        return prepare_result(df)

    def int_to_date(self, col_name: str) -> FrameType:
        df = impl_int_to_date(self._df, col_name)
        return prepare_result(df)

    def date_to_int(self, col_name: str, out: Optional[str] = None) -> FrameType:
        out = col_name if out is None else out
        df = impl_date_to_int(self._df, col_name, out)
        return prepare_result(df)

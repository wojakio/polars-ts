from typing import Generic, Optional
import polars as pl
from polars.type_aliases import IntoExpr

from .sf import SeriesFrame
from .sf_helper import prepare_result
from .utils import parse_into_expr
from .convert_helper import impl_convert, impl_construct_closure

from .types import FrameType

__NAMESPACE = "convert"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class ConvertFrame(SeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType):
        super().__init__(df)

    def construct_closure(
        self,
        max_iterations: int = 1000,
    ) -> FrameType:
        df = impl_construct_closure(
            self._df,
            max_iterations,
        )

        return prepare_result(df)

    def convert(
        self,
        target_unit: IntoExpr,
        conversion_matrix: FrameType,
        value: IntoExpr = pl.col("value"),
        value_unit: Optional[IntoExpr] = None,
        is_multi_dim: bool = False,
    ) -> FrameType:
        target_unit = parse_into_expr(target_unit).cast(pl.Categorical)
        value = parse_into_expr(value)

        if value_unit is None:
            value_unit = f"{value.meta.output_name()}_unit"
            if value_unit not in self._df.columns:
                raise ValueError(f"Missing unit column: '{value_unit}'")

        value_unit = parse_into_expr(value_unit).cast(pl.Categorical)

        df = impl_convert(
            self._df,
            target_unit,
            conversion_matrix,
            value,
            value_unit,
            is_multi_dim,
        )

        return prepare_result(df)

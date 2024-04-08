import polars as pl

from .sf import SeriesFrame

__NAMESPACE = "tsf"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class TimeSeriesFrame(SeriesFrame):
    def __init__(self, df: pl.LazyFrame):
        if "time" not in df.columns:
            raise ValueError("Missing column: ", "time")

        super().__init__(df)

import polars as pl

from .sf import SeriesFrame
from .futures_helper.roll_calendar import create_roll_calendar_helper

__NAMESPACE = "future"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class FuturesFrame(SeriesFrame):
    def __init__(self, df: pl.LazyFrame) -> None:
        super().__init__(df)

    def create_roll_calendar(
        self,
        roll_config: pl.LazyFrame,
        security_expiries: pl.LazyFrame,
        include_debug: bool = False
    ) -> pl.LazyFrame:

        self._df = create_roll_calendar_helper(
            roll_config,
            security_expiries,
            include_debug
        )

        return self.result_df

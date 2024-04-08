from datetime import datetime

import polars as pl

from .sf import SeriesFrame
from .futures_helper.roll_calendar import create_roll_calendars, load_roll_config

__NAMESPACE = "future"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class FuturesFrame(SeriesFrame):
    def __init__(self, df: pl.LazyFrame):
        super().__init__(df)

    def create_roll_calendar(
        self, filename: str, start_date: datetime, end_date: datetime
    ) -> pl.LazyFrame:
        roll_config = load_roll_config(filename)
        roll_calendar = create_roll_calendars(roll_config, start_date, end_date)
        return roll_calendar

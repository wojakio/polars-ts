from typing import Generic
import polars as pl

from .sf import SeriesFrame
from .sf_helper import prepare_result
from .futures_helper import (
    impl_create_roll_calendar,
    impl_prepare_unadjusted_for_stitching,
    impl_stitch_panama_backwards,
)
from .types import FrameType

__NAMESPACE = "future"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class FuturesFrame(SeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType) -> None:
        super().__init__(df)

    def create_roll_calendar(
        self,
        roll_config: FrameType,
        security_expiries: FrameType,
        include_debug: bool = False,
    ) -> FrameType:
        df = impl_create_roll_calendar(roll_config, security_expiries, include_debug)
        return prepare_result(df)

    def prepare_unadjusted_for_stitching(
        self,
        roll_calendar: FrameType,
        price_universe: FrameType,
    ) -> FrameType:
        df = impl_prepare_unadjusted_for_stitching(
            self._df, roll_calendar, price_universe
        )
        return prepare_result(df)

    def stitch_panama_backwards(self) -> FrameType:
        df = impl_stitch_panama_backwards(self._df)
        return prepare_result(df)

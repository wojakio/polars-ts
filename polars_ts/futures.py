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

    def prepare_unadjusted_for_stitching(
        self,
        roll_calendar: pl.LazyFrame,
        price_universe: pl.LazyFrame,
    ) -> pl.LazyFrame:

        far_contract_prices = (
            pl.LazyFrame()
            .dummymkt.fetch_roll_calendar_prices(roll_calendar, price_universe)
            .select("asset", time="roll_date", instrument_id="near_contract", next_instrument_id="far_contract", next_value="far_price")
        )

        self._df = (
            roll_calendar
            .select(
                "asset",
                instrument_id="near_contract",
                instrument_start_date=pl.col("roll_date").shift(1, fill_value=pl.min("roll_date").dt.offset_by("-5y")).over("asset"),
                instrument_end_date="roll_date",
            )
            .join(price_universe, on=["asset", "instrument_id"], how="left")
            .join(far_contract_prices, on=["time", "asset", "instrument_id"], how="left")
            .filter(pl.col("time").is_between("instrument_start_date", "instrument_end_date", "right").over("instrument_id"))
            .select(pl.exclude("instrument_start_date", "instrument_end_date"))
        )

        return self._df

    def stitch_panama_backwards(self) -> pl.LazyFrame:

        self._df = (
            self._df
            .with_columns(
                roll_adj=(
                    pl.when(pl.col("next_value").is_not_null())
                    .then(pl.col("next_value") - pl.col("value"))
                    .otherwise(0.)
                )
            )
            .with_columns(
                cum_adj=(
                    pl.col("roll_adj")
                    .reverse().cum_sum().reverse()
                    .over("asset")
                )
            )
            .with_columns(adj=pl.col("value") + pl.col("cum_adj"))

            # cleanup
            .select("time", "asset", "instrument_id", unadjusted="value", panama_backwards="adj")
        )

        return self.result_df
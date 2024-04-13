import polars as pl

from .sf import SeriesFrame

__NAMESPACE = "loader"

@pl.api.register_lazyframe_namespace(__NAMESPACE)
class LoaderFrame(SeriesFrame):
    def __init__(self, df: pl.LazyFrame) -> None:
        super().__init__(df)

    def csv(self, filename: str) -> pl.LazyFrame:
        df = (
            pl.scan_csv(filename, comment_prefix="//")
            .with_columns(pl.col(pl.String).cast(pl.Categorical))
        )

        return df

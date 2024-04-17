from typing import Any, List
import polars as pl

repr_overloaded: List[int] = []

if len(repr_overloaded) == 0:
    oldrepr = pl.DataFrame._repr_html_

    def newrepr(self, **kwargs: Any) -> str:
        time_col = ["time"] if "time" in self.columns else []
        category_cols = [
            c
            for c, dtype in zip(self.columns, self.dtypes)
            if dtype in [pl.Categorical, pl.Enum]
        ]

        cols = time_col + category_cols

        return oldrepr(
            self.select(*[pl.col(c) for c in cols], pl.exclude(cols)), **kwargs
        )

    pl.DataFrame._repr_html_ = newrepr  # type: ignore[method-assign]

    repr_overloaded = [1]

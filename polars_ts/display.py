from typing import Any
import polars as pl

repr_overloaded = []

if len(repr_overloaded) == 0:
    oldrepr=pl.DataFrame._repr_html_
    def newrepr(self, **kwargs: Any) -> str:
        time_col = ['time'] if 'time' in self.columns else []
        category_cols = [
            c for c, dtype in zip(self.columns, self.dtypes)
            if dtype == pl.Categorical
        ]

        cols = time_col + category_cols
 
        return oldrepr(
            self.select(
                *[pl.col(c) for c in cols],
                pl.exclude(cols)
            ),
            **kwargs
        )

    pl.DataFrame._repr_html_=newrepr

    repr_overloaded = [1]
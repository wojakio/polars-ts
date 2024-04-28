def debug_me():

    import os
    os.environ["POLARS_VERBOSE"]="1"

    from datetime import datetime, timedelta
    import polars as pl
    import polars_ts as ts #noqa

    print("DBG: imports completed. Running script...")

    params = pl.LazyFrame(
        [
            pl.Series("method", ["arithmetic", "geometric", "fractional"] * 2),
            # pl.Series("method", ["arithmetic", "arithmetic", "arithmetic"] * 2),
            pl.Series("n", [1, 1, 1, 2, 2, 2]),
        ]
    ).with_columns(pl.col(pl.String).cast(pl.Categorical))

    t = datetime(2024, 1, 1).date()
    steps = list(range(0, 100, 5))
    nrows = len(steps)

    df = pl.LazyFrame(
        [
            pl.Series(
                "time", [t + timedelta(days=days + 1) for days in range(0, nrows * 2, 2)]
            ),
            pl.Series("value", list(range(0, nrows)), dtype=pl.Float64),
        ]
    )

    result_df = df.mathx.diff(params).collect()

    print(result_df)

if __name__ == "__main__":
    print("DBG: about to call fn...")
    debug_me()
    print("DBG: finished debugging!")


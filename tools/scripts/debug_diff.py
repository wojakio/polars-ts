def debug_me():

    import os
    os.environ["POLARS_VERBOSE"]="1"

    import polars as pl
    import polars_ts as ts #noqa

    print("DBG: imports completed. Running script...")

    df = pl.DataFrame([
        pl.Series("name", ["A", "A", "A", "B", "B", "B"]),
        pl.Series("value", [None, 1, 2, None, 3, 5]),

    ])

    (
        df
        .group_by("name", maintain_order=True)
        .agg(
            pl.exclude("value"),
            ts.expr.sf.handle_null_custom("value", pl.lit("forward"), pl.lit(0.0)),
        )
        .explode(pl.col("value"))
    )

    # df = pl.LazyFrame([
    #     pl.Series("a", ["A"] * 5, dtype=pl.Categorical),
    #     pl.Series("value", [None, None, 1, 1, 1], dtype=pl.Float64),

    # ])

    # params = pl.LazyFrame([
    #     pl.Series("a", ["A"], dtype=pl.Categorical),
    #     pl.Series("null_strategy", ["drop_all"], dtype=pl.Categorical)
    # ])

    # (
    #     df
    #     .sf.handle_null(params)
    #     .collect()
    # )
    # params = pl.LazyFrame(
    #     [
    #         pl.Series("method", ["arithmetic", "geometric", "fractional"] * 2),
    #         # pl.Series("method", ["arithmetic", "arithmetic", "arithmetic"] * 2),
    #         pl.Series("n", [1, 1, 1, 2, 2, 2]),
    #     ]
    # ).with_columns(pl.col(pl.String).cast(pl.Categorical))

    # t = datetime(2024, 1, 1).date()
    # steps = list(range(0, 100, 5))
    # nrows = len(steps)

    # df = pl.LazyFrame(
    #     [
    #         pl.Series(
    #             "time", [t + timedelta(days=days + 1) for days in range(0, nrows * 2, 2)]
    #         ),
    #         pl.Series("value", list(range(0, nrows)), dtype=pl.Float64),
    #     ]
    # )

    # result_df = df.mathx.diff(params).collect()

    # print(result_df)

if __name__ == "__main__":
    print("DBG: about to call fn...")
    debug_me()
    print("DBG: finished debugging!")


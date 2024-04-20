def debug_me():

    import os
    os.environ["POLARS_VERBOSE"]="1"

    from datetime import datetime, timedelta
    import polars as pl
    import polars_ts as ts #noqa

    print("DBG: imports completed. Running script...")

    holiday_calendars = (
        pl.DataFrame({
            "name": ["hols_A", "hols_B", "hols_C"],
            "dates": [
                [datetime(2000,1,4), datetime(2000,1,19), datetime(2000,1,5),datetime(2000,1,20), datetime(2000,1,21)],
                [datetime(2001,1,1), datetime(2001,1,2), datetime(2001,1,3),datetime(2001,1,4), datetime(2001,1,5)],
                [datetime(2000,1,1) + timedelta(days=t) for t in range(365*5)],
            ]
        })
        .with_columns(pl.col("dates").cast(pl.List(pl.Date)))
    )

    # A - regular use case
    # B - all dates in span are holidays
    # C - no holiday calendar
    # D - date range is 0 days
    # E - nothing is a holiday

    df = (
        pl.LazyFrame([
            pl.Series("X", ["A", "B", "C", "D", "E"]),
            pl.Series("start", [datetime(2000,1,1), datetime(2001,1,1), datetime(2002,1,1), datetime(2003,1,1), datetime(2002,1,1)]),
            pl.Series("end", [datetime(2000,2,1), datetime(2001,1,5), datetime(2002,2,2), datetime(2003,1,1), datetime(2003,1,1)]),
            pl.Series("holiday_cdr", [["hols_A"], ["hols_A", "hols_B", "hols_C"], None, ["hols_C"], []])
        ])
        .with_columns(
            pl.col("start").cast(pl.Date),
            pl.col("end").cast(pl.Date),
        )
        .sf.join_on_list_items(
            holiday_calendars.lazy(),
            left_on=pl.col("holiday_cdr"),
            right_on=pl.col("name"),
            how="left",
            flatten=True,
            then_unique=False,
            then_sort=False
        )
        .drop("holiday_cdr")
        .with_columns(weekends=pl.lit([6,7]))
        # .with_columns(dates=None)
    )

    input = pl.concat([df, df, df, df, df])

    # (
    #     input
    #     .datetime_ranges("start", "end")
    #     .collect()
    # )

    x = (
        input
        .time.ranges("start", "end", "dates", "weekends")
        .collect()
    )

    print(x)


if __name__ == "__main__":
    print("DBG: about to call fn...")
    debug_me()
    print("DBG: finished debugging!")


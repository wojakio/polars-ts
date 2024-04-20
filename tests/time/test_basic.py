from datetime import datetime
import polars as pl
import polars_ts  # noqa
import pytest


@pytest.fixture
def holiday_calendars() -> pl.LazyFrame:
    df = pl.LazyFrame(
        [
            pl.Series(
                "name",
                ["A", "B", "C", "D", "LIST_NULL", "EMPTY", "NULL"],
                dtype=pl.String,
            ),
            pl.Series(
                "holidays",
                [
                    [datetime(2000, 1, 3), datetime(2000, 1, 4), datetime(2000, 1, 5)],
                    [
                        datetime(2000, 1, 4),
                        datetime(2000, 1, 5),
                        datetime(2000, 1, 6),
                        datetime(2000, 1, 7),
                        datetime(2000, 1, 8),
                    ],
                    [datetime(2000, 1, 9)],
                    [datetime(2000, 1, 10), None],
                    [None],
                    None,
                    [],
                ],
            ).cast(pl.List(pl.Date)),
            pl.Series("iso_weekends", [[6]] + [[6, 7]] * 6),
        ]
    )

    return df


def test_basic(holiday_calendars):
    inputs = (
        pl.from_records(
            # 2000,1,1 = saturday, 2000,1,2 = sunday, 2000-1-10 = monday
            data=[
                [datetime(2000, 1, 1), datetime(2000, 1, 10), ["A", "B", "C", "D"]],
                [datetime(2000, 1, 1), datetime(2000, 1, 17), ["A", "B", "C", "D"]],
                [datetime(2000, 1, 4), datetime(2000, 1, 4), ["A"]],
                [datetime(2000, 1, 4), datetime(2000, 1, 4), ["B"]],
                [datetime(2000, 1, 4), datetime(2000, 1, 4), ["C"]],
                [datetime(2000, 1, 4), datetime(2000, 1, 4), ["D"]],
                [datetime(2000, 1, 1), datetime(2000, 1, 5), ["LIST_NULL"]],
                [datetime(2000, 1, 1), datetime(2000, 1, 5), ["EMPTY"]],
                [datetime(2000, 1, 1), datetime(2000, 1, 5), ["NULL"]],  # panic
            ],
            schema=["start_date", "end_date", "calendar"],
            orient="row",
        )
        .with_columns(pl.col(pl.DATETIME_DTYPES).cast(pl.Date))
        .lazy()
        .sf.join_on_list_items(
            holiday_calendars, left_on="calendar", right_on="name", how="left"
        )
        .with_columns(
            pl.when(pl.col("calendar").list.first() == "EMPTY")
            .then(pl.lit([]))
            .when(pl.col("calendar").list.first() == "NULL")
            .then(pl.lit(None))
            .otherwise(pl.col("holidays"))
            .alias("holidays")
        )
    )

    result = inputs.time.ranges(
        "start_date", "end_date", "holidays", "iso_weekends"
    ).collect()

    expected_result = (
        pl.LazyFrame(
            [
                pl.Series(
                    "start_date",
                    [
                        datetime(2000, 1, 1),
                        datetime(2000, 1, 1),
                        datetime(2000, 1, 4),
                        datetime(2000, 1, 4),
                        datetime(2000, 1, 4),
                        datetime(2000, 1, 4),
                        datetime(2000, 1, 1),
                        datetime(2000, 1, 1),
                        datetime(2000, 1, 1),
                    ],
                    dtype=pl.Date,
                ),
                pl.Series(
                    "end_date",
                    [
                        datetime(2000, 1, 10),
                        datetime(2000, 1, 17),
                        datetime(2000, 1, 4),
                        datetime(2000, 1, 4),
                        datetime(2000, 1, 4),
                        datetime(2000, 1, 4),
                        datetime(2000, 1, 5),
                        datetime(2000, 1, 5),
                        datetime(2000, 1, 5),
                    ],
                    dtype=pl.Date,
                ),
                pl.Series(
                    "calendar",
                    [
                        ["A", "B", "C", "D"],
                        ["A", "B", "C", "D"],
                        ["A"],
                        ["B"],
                        ["C"],
                        ["D"],
                        ["LIST_NULL"],
                        ["EMPTY"],
                        ["NULL"],
                    ],
                    dtype=pl.List(pl.String),
                ),
                pl.Series(
                    "holidays",
                    [
                        [
                            None,
                            datetime(2000, 1, 3),
                            datetime(2000, 1, 4),
                            datetime(2000, 1, 5),
                            datetime(2000, 1, 6),
                            datetime(2000, 1, 7),
                            datetime(2000, 1, 8),
                            datetime(2000, 1, 9),
                            datetime(2000, 1, 10),
                        ],
                        [
                            None,
                            datetime(2000, 1, 3),
                            datetime(2000, 1, 4),
                            datetime(2000, 1, 5),
                            datetime(2000, 1, 6),
                            datetime(2000, 1, 7),
                            datetime(2000, 1, 8),
                            datetime(2000, 1, 9),
                            datetime(2000, 1, 10),
                        ],
                        [
                            datetime(2000, 1, 3),
                            datetime(2000, 1, 4),
                            datetime(2000, 1, 5),
                        ],
                        [
                            datetime(2000, 1, 4),
                            datetime(2000, 1, 5),
                            datetime(2000, 1, 6),
                            datetime(2000, 1, 7),
                            datetime(2000, 1, 8),
                        ],
                        [datetime(2000, 1, 9)],
                        [None, datetime(2000, 1, 10)],
                        [None],
                        [],
                        None,
                    ],
                ),
                pl.Series(
                    "iso_weekends",
                    [
                        [6, 7],
                        [6, 7],
                        [6],
                        [6, 7],
                        [6, 7],
                        [6, 7],
                        [6, 7],
                        [6, 7],
                        [6, 7],
                    ],
                ),
                pl.Series(
                    "ranges",
                    [
                        [],
                        [
                            datetime(2000, 1, 11),
                            datetime(2000, 1, 12),
                            datetime(2000, 1, 13),
                            datetime(2000, 1, 14),
                            datetime(2000, 1, 17),
                        ],
                        [],
                        [],
                        [datetime(2000, 1, 4)],
                        [datetime(2000, 1, 4)],
                        [
                            datetime(2000, 1, 3),
                            datetime(2000, 1, 4),
                            datetime(2000, 1, 5),
                        ],
                        [
                            datetime(2000, 1, 3),
                            datetime(2000, 1, 4),
                            datetime(2000, 1, 5),
                        ],
                        None,
                    ],
                ),
            ]
        )
        .with_columns(pl.col(pl.DATETIME_DTYPES).cast(pl.Date))
        .collect()
    )

    assert result.equals(expected_result)

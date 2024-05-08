from datetime import datetime, timedelta
import polars as pl
import polars_ts  # noqa
import pytest


@pytest.fixture
def df() -> pl.LazyFrame:
    t = datetime(2024, 1, 1, 0)
    nrows = 3
    result = pl.LazyFrame(
        {
            "time": [t]
            + [t + timedelta(days=days + 1) for days in range(0, nrows * 2, 2)],
            "flt1": [0] + [2.18 + i for i in range(nrows)],
            "text": ["X"] + [f"stext {i}" for i in range(nrows)],
            "catsa": ["CA"] + [f"A{i}" for i in range(nrows)],
            "catsb": ["CB"] + [f"B{i % 3}" for i in range(nrows)],
            "flt2": [0, 0] + [1.39 + i for i in range(nrows - 1)],
        },
        schema={
            "time": pl.Datetime,
            "text": pl.String,
            "catsa": pl.Categorical,
            "catsb": pl.Categorical,
            "flt1": pl.Float64,
            "flt2": pl.Float32,
        },
    )

    return result


@pytest.fixture
def df_other() -> pl.LazyFrame:
    result = pl.LazyFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2025, 1, 1)],
            "catsa": ["CA", "CA"],
            "mean": [999, 888],
        },
        schema={"time": pl.Datetime, "catsa": pl.Categorical, "mean": pl.Float32},
    )

    return result


def test_resample_categories(df):
    time_axis = pl.datetime_ranges(
        df.collect()["time"].min(), df.collect()["time"].max(), eager=True
    ).explode()

    align_none = df.rs.resample_categories(time_axis, closed="none").collect()
    align_left = df.rs.resample_categories(time_axis, closed="left").collect()
    align_right = df.rs.resample_categories(time_axis, closed="right").collect()
    align_both = df.rs.resample_categories(time_axis, closed="both").collect()

    expected_none = pl.DataFrame(
        [
            pl.Series(
                "time",
                [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2, 0),
                    datetime(2024, 1, 2, 0),
                    datetime(2024, 1, 2, 0),
                    datetime(2024, 1, 2, 0),
                    datetime(2024, 1, 3, 0),
                    datetime(2024, 1, 3, 0),
                    datetime(2024, 1, 3, 0),
                    datetime(2024, 1, 3, 0),
                    datetime(2024, 1, 4, 0),
                    datetime(2024, 1, 4, 0),
                    datetime(2024, 1, 4, 0),
                    datetime(2024, 1, 4, 0),
                    datetime(2024, 1, 5, 0),
                    datetime(2024, 1, 5, 0),
                    datetime(2024, 1, 5, 0),
                    datetime(2024, 1, 5, 0),
                    datetime(2024, 1, 6, 0),
                    datetime(2024, 1, 6, 0),
                    datetime(2024, 1, 6, 0),
                    datetime(2024, 1, 6, 0, 0),
                ],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series(
                "catsa",
                [
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "catsb",
                [
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
        ]
    )

    expected_left = pl.DataFrame(
        [
            pl.Series(
                "time",
                [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2, 0),
                    datetime(2024, 1, 2, 0),
                    datetime(2024, 1, 3, 0),
                    datetime(2024, 1, 3, 0),
                    datetime(2024, 1, 4, 0),
                    datetime(2024, 1, 4, 0),
                    datetime(2024, 1, 4, 0),
                    datetime(2024, 1, 5, 0),
                    datetime(2024, 1, 5, 0),
                    datetime(2024, 1, 5, 0),
                    datetime(2024, 1, 6, 0),
                    datetime(2024, 1, 6, 0),
                    datetime(2024, 1, 6, 0),
                    datetime(2024, 1, 6, 0, 0),
                ],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series(
                "catsa",
                [
                    "CA",
                    "CA",
                    "A0",
                    "CA",
                    "A0",
                    "CA",
                    "A0",
                    "A1",
                    "CA",
                    "A0",
                    "A1",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "catsb",
                [
                    "CB",
                    "CB",
                    "B0",
                    "CB",
                    "B0",
                    "CB",
                    "B0",
                    "B1",
                    "CB",
                    "B0",
                    "B1",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
        ]
    )

    expected_right = pl.DataFrame(
        [
            pl.Series(
                "time",
                [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2, 0),
                    datetime(2024, 1, 2, 0),
                    datetime(2024, 1, 2, 0),
                    datetime(2024, 1, 3, 0),
                    datetime(2024, 1, 3, 0),
                    datetime(2024, 1, 4, 0),
                    datetime(2024, 1, 4, 0),
                    datetime(2024, 1, 5, 0),
                    datetime(2024, 1, 6, 0, 0),
                ],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series(
                "catsa",
                [
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "A0",
                    "A1",
                    "A2",
                    "A1",
                    "A2",
                    "A1",
                    "A2",
                    "A2",
                    "A2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "catsb",
                [
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "B0",
                    "B1",
                    "B2",
                    "B1",
                    "B2",
                    "B1",
                    "B2",
                    "B2",
                    "B2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
        ]
    )

    expected_both = pl.DataFrame(
        [
            pl.Series(
                "time",
                [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2, 0),
                    datetime(2024, 1, 4, 0),
                    datetime(2024, 1, 6, 0, 0),
                ],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series(
                "catsa",
                ["CA", "A0", "A1", "A2"],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "catsb",
                ["CB", "B0", "B1", "B2"],
                dtype=pl.Categorical(ordering="physical"),
            ),
        ]
    )

    assert align_none.equals(expected_none)
    assert align_left.equals(expected_left)
    assert align_right.equals(expected_right)
    assert align_both.equals(expected_both)


def test_resample_categories_no_categories(df):
    time_axis = pl.datetime_ranges(
        df.collect()["time"].min(), df.collect()["time"].max(), eager=True
    ).explode()

    time_axis = pl.datetime_ranges(
        df.collect()["time"].min(), df.collect()["time"].max(), eager=True
    ).explode()

    df = df.select("time", "flt1")
    align_none = df.rs.resample_categories(time_axis, closed="none").collect()
    align_left = df.rs.resample_categories(time_axis, closed="left").collect()
    align_right = df.rs.resample_categories(time_axis, closed="right").collect()
    align_both = df.rs.resample_categories(time_axis, closed="both").collect()

    expected_df = pl.DataFrame(
        [
            pl.Series(
                "time",
                [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2, 0),
                    datetime(2024, 1, 3, 0),
                    datetime(2024, 1, 4, 0),
                    datetime(2024, 1, 5, 0),
                    datetime(2024, 1, 6, 0, 0),
                ],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
        ]
    )

    assert align_none.equals(expected_df)
    assert align_left.equals(expected_df)
    assert align_right.equals(expected_df)
    assert align_both.equals(expected_df)


def test_align_to_time(df):
    time_axis = pl.datetime_ranges(
        df.collect()["time"].min(), df.collect()["time"].max(), eager=True
    ).explode()

    align_none = df.rs.align_to_time(
        time_axis, closed="none", null_strategy="ignore"
    ).collect()
    align_left = df.rs.align_to_time(
        time_axis, closed="left", null_strategy="ignore"
    ).collect()
    align_right = df.rs.align_to_time(
        time_axis, closed="right", null_strategy="ignore"
    ).collect()
    align_both = df.rs.align_to_time(
        time_axis, closed="both", null_strategy="ignore"
    ).collect()

    expected_none = pl.DataFrame(
        [
            pl.Series(
                "time",
                [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 2, 0, 0),
                    datetime(2024, 1, 2, 0, 0),
                    datetime(2024, 1, 2, 0, 0),
                    datetime(2024, 1, 2, 0, 0),
                    datetime(2024, 1, 3, 0, 0),
                    datetime(2024, 1, 3, 0, 0),
                    datetime(2024, 1, 3, 0, 0),
                    datetime(2024, 1, 3, 0, 0),
                    datetime(2024, 1, 4, 0, 0),
                    datetime(2024, 1, 4, 0, 0),
                    datetime(2024, 1, 4, 0, 0),
                    datetime(2024, 1, 4, 0, 0),
                    datetime(2024, 1, 5, 0, 0),
                    datetime(2024, 1, 5, 0, 0),
                    datetime(2024, 1, 5, 0, 0),
                    datetime(2024, 1, 5, 0, 0),
                    datetime(2024, 1, 6, 0, 0),
                    datetime(2024, 1, 6, 0, 0),
                    datetime(2024, 1, 6, 0, 0),
                    datetime(2024, 1, 6, 0, 0),
                ],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series(
                "text",
                [
                    "X",
                    None,
                    None,
                    None,
                    None,
                    "stext 0",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "stext 1",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "stext 2",
                ],
                dtype=pl.String,
            ),
            pl.Series(
                "catsa",
                [
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "catsb",
                [
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "flt1",
                [
                    0.0,
                    None,
                    None,
                    None,
                    None,
                    2.18,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    3.18,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    4.18,
                ],
                dtype=pl.Float64,
            ),
            pl.Series(
                "flt2",
                [
                    0.0,
                    None,
                    None,
                    None,
                    None,
                    0.0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    1.3899999856948853,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    2.390000104904175,
                ],
                dtype=pl.Float32,
            ),
        ]
    )

    expected_left = pl.DataFrame(
        [
            pl.Series(
                "time",
                [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 2, 0, 0),
                    datetime(2024, 1, 2, 0, 0),
                    datetime(2024, 1, 3, 0, 0),
                    datetime(2024, 1, 3, 0, 0),
                    datetime(2024, 1, 4, 0, 0),
                    datetime(2024, 1, 4, 0, 0),
                    datetime(2024, 1, 4, 0, 0),
                    datetime(2024, 1, 5, 0, 0),
                    datetime(2024, 1, 5, 0, 0),
                    datetime(2024, 1, 5, 0, 0),
                    datetime(2024, 1, 6, 0, 0),
                    datetime(2024, 1, 6, 0, 0),
                    datetime(2024, 1, 6, 0, 0),
                    datetime(2024, 1, 6, 0, 0),
                ],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series(
                "text",
                [
                    "X",
                    None,
                    "stext 0",
                    None,
                    None,
                    None,
                    None,
                    "stext 1",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "stext 2",
                ],
                dtype=pl.String,
            ),
            pl.Series(
                "catsa",
                [
                    "CA",
                    "CA",
                    "A0",
                    "CA",
                    "A0",
                    "CA",
                    "A0",
                    "A1",
                    "CA",
                    "A0",
                    "A1",
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "catsb",
                [
                    "CB",
                    "CB",
                    "B0",
                    "CB",
                    "B0",
                    "CB",
                    "B0",
                    "B1",
                    "CB",
                    "B0",
                    "B1",
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "flt1",
                [
                    0.0,
                    None,
                    2.18,
                    None,
                    None,
                    None,
                    None,
                    3.18,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    4.18,
                ],
                dtype=pl.Float64,
            ),
            pl.Series(
                "flt2",
                [
                    0.0,
                    None,
                    0.0,
                    None,
                    None,
                    None,
                    None,
                    1.3899999856948853,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    2.390000104904175,
                ],
                dtype=pl.Float32,
            ),
        ]
    )

    expected_right = pl.DataFrame(
        [
            pl.Series(
                "time",
                [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 2, 0, 0),
                    datetime(2024, 1, 2, 0, 0),
                    datetime(2024, 1, 2, 0, 0),
                    datetime(2024, 1, 3, 0, 0),
                    datetime(2024, 1, 3, 0, 0),
                    datetime(2024, 1, 4, 0, 0),
                    datetime(2024, 1, 4, 0, 0),
                    datetime(2024, 1, 5, 0, 0),
                    datetime(2024, 1, 6, 0, 0),
                ],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series(
                "text",
                [
                    "X",
                    None,
                    None,
                    None,
                    "stext 0",
                    None,
                    None,
                    None,
                    None,
                    "stext 1",
                    None,
                    None,
                    "stext 2",
                ],
                dtype=pl.String,
            ),
            pl.Series(
                "catsa",
                [
                    "CA",
                    "A0",
                    "A1",
                    "A2",
                    "A0",
                    "A1",
                    "A2",
                    "A1",
                    "A2",
                    "A1",
                    "A2",
                    "A2",
                    "A2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "catsb",
                [
                    "CB",
                    "B0",
                    "B1",
                    "B2",
                    "B0",
                    "B1",
                    "B2",
                    "B1",
                    "B2",
                    "B1",
                    "B2",
                    "B2",
                    "B2",
                ],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "flt1",
                [
                    0.0,
                    None,
                    None,
                    None,
                    2.18,
                    None,
                    None,
                    None,
                    None,
                    3.18,
                    None,
                    None,
                    4.18,
                ],
                dtype=pl.Float64,
            ),
            pl.Series(
                "flt2",
                [
                    0.0,
                    None,
                    None,
                    None,
                    0.0,
                    None,
                    None,
                    None,
                    None,
                    1.3899999856948853,
                    None,
                    None,
                    2.390000104904175,
                ],
                dtype=pl.Float32,
            ),
        ]
    )

    expected_both = pl.DataFrame(
        [
            pl.Series(
                "time",
                [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 2, 0, 0),
                    datetime(2024, 1, 4, 0, 0),
                    datetime(2024, 1, 6, 0, 0),
                ],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series("text", ["X", "stext 0", "stext 1", "stext 2"], dtype=pl.String),
            pl.Series(
                "catsa",
                ["CA", "A0", "A1", "A2"],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series(
                "catsb",
                ["CB", "B0", "B1", "B2"],
                dtype=pl.Categorical(ordering="physical"),
            ),
            pl.Series("flt1", [0.0, 2.18, 3.18, 4.18], dtype=pl.Float64),
            pl.Series(
                "flt2",
                [0.0, 0.0, 1.3899999856948853, 2.390000104904175],
                dtype=pl.Float32,
            ),
        ]
    )

    assert align_none.equals(expected_none)
    assert align_left.equals(expected_left)
    assert align_right.equals(expected_right)
    assert align_both.equals(expected_both)


def test_align_values(df, df_other):
    align_self = df.rs.align_values(df).collect()
    assert align_self.equals(df.collect())

    align_df_to_other_lhs = df.rs.align_values(df_other).collect()
    expected_align_df_to_other_lhs = pl.DataFrame(
        [
            pl.Series(
                "time",
                [datetime(2024, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series("text", ["X", None], dtype=pl.String),
            pl.Series("catsa", ["CA", "CA"], dtype=pl.Categorical(ordering="physical")),
            pl.Series("catsb", ["CB", None], dtype=pl.Categorical(ordering="physical")),
            pl.Series("flt1", [0.0, None], dtype=pl.Float64),
            pl.Series("flt2", [0.0, None], dtype=pl.Float32),
        ]
    )
    assert align_df_to_other_lhs.equals(expected_align_df_to_other_lhs)

    align_other_to_df_lhs = df.rs.align_values(df_other).collect()
    expected_align_other_to_df_lhs = pl.DataFrame(
        [
            pl.Series(
                "time",
                [datetime(2024, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series("text", ["X", None], dtype=pl.String),
            pl.Series("catsa", ["CA", "CA"], dtype=pl.Categorical(ordering="physical")),
            pl.Series("catsb", ["CB", None], dtype=pl.Categorical(ordering="physical")),
            pl.Series("flt1", [0.0, None], dtype=pl.Float64),
            pl.Series("flt2", [0.0, None], dtype=pl.Float32),
        ]
    )
    assert align_other_to_df_lhs.equals(expected_align_other_to_df_lhs)

    align_df_to_other_rhs = df.rs.align_values(df_other, retain_values="rhs").collect()
    expected_align_df_to_other_rhs = pl.DataFrame(
        [
            pl.Series(
                "time",
                [datetime(2024, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series("catsa", ["CA", "CA"], dtype=pl.Categorical(ordering="physical")),
            pl.Series("catsb", ["CB", None], dtype=pl.Categorical(ordering="physical")),
            pl.Series("mean", [999.0, 888.0], dtype=pl.Float32),
        ]
    )
    assert align_df_to_other_rhs.equals(expected_align_df_to_other_rhs)

    align_other_to_df_rhs = df.rs.align_values(df_other, retain_values="rhs").collect()
    expected_align_other_to_df_rhs = pl.DataFrame(
        [
            pl.Series(
                "time",
                [datetime(2024, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series("catsa", ["CA", "CA"], dtype=pl.Categorical(ordering="physical")),
            pl.Series("catsb", ["CB", None], dtype=pl.Categorical(ordering="physical")),
            pl.Series("mean", [999.0, 888.0], dtype=pl.Float32),
        ]
    )
    assert align_other_to_df_rhs.equals(expected_align_other_to_df_rhs)

    align_df_to_other_both = df.rs.align_values(
        df_other, retain_values="both"
    ).collect()
    expected_align_df_to_other_both = pl.DataFrame(
        [
            pl.Series(
                "time",
                [datetime(2024, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series("text", ["X", None], dtype=pl.String),
            pl.Series("catsa", ["CA", "CA"], dtype=pl.Categorical(ordering="physical")),
            pl.Series("catsb", ["CB", None], dtype=pl.Categorical(ordering="physical")),
            pl.Series("flt1", [0.0, None], dtype=pl.Float64),
            pl.Series("flt2", [0.0, None], dtype=pl.Float32),
            pl.Series("mean", [999.0, 888.0], dtype=pl.Float32),
        ]
    )
    assert align_df_to_other_both.equals(expected_align_df_to_other_both)

    align_other_to_df_both = df.rs.align_values(
        df_other, retain_values="both"
    ).collect()
    expected_align_other_to_df_both = pl.DataFrame(
        [
            pl.Series(
                "time",
                [datetime(2024, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 0)],
                dtype=pl.Datetime(time_unit="us", time_zone=None),
            ),
            pl.Series("text", ["X", None], dtype=pl.String),
            pl.Series("catsa", ["CA", "CA"], dtype=pl.Categorical(ordering="physical")),
            pl.Series("catsb", ["CB", None], dtype=pl.Categorical(ordering="physical")),
            pl.Series("flt1", [0.0, None], dtype=pl.Float64),
            pl.Series("flt2", [0.0, None], dtype=pl.Float32),
            pl.Series("mean", [999.0, 888.0], dtype=pl.Float32),
        ]
    )
    assert align_other_to_df_both.equals(expected_align_other_to_df_both)

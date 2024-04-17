from datetime import datetime, timedelta
import polars as pl
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

from datetime import datetime, timedelta
import polars as pl
import pytest
import polars_ts  # noqa


@pytest.fixture
def df_map() -> pl.LazyFrame:
    df = pl.LazyFrame(
        {
            "date_key": [
                (datetime(2000, 1, 1) + timedelta(days=i - 1)).date()
                for i in range(1, 7)
            ],
            "int_key": range(1, 7),
            "str_key": [chr(ord("z") - i + 1) for i in range(1, 7)],
            "B": [[], [1], [1, 2], [2, 3, 4], [5], [6, 7, 8, 9, 10]],
        }
    )

    return df


def test_int_key(df_map):
    df = pl.LazyFrame(
        {
            "mylist": [
                [1],
                [1],
                [2, 3],
                [],
                None,
                [2, 3, 6],
                [2, 2, 2],
                [9],
                [1, 9],
                [2, 9],
            ]
        }
    )

    result = df.sf.join_on_list_items(
        df_map, left_on="mylist", right_on="int_key", how="left"
    ).collect()

    expected_result = pl.DataFrame(
        [
            pl.Series(
                "mylist",
                [[1], [1], [2, 3], [], None, [2, 3, 6], [2, 2, 2], [9], [1, 9], [2, 9]],
                dtype=pl.List(pl.Int64),
            ),
            pl.Series(
                "date_key",
                [
                    [datetime(2000, 1, 1)],
                    [datetime(2000, 1, 1)],
                    [datetime(2000, 1, 2), datetime(2000, 1, 3)],
                    None,
                    None,
                    [datetime(2000, 1, 2), datetime(2000, 1, 3), datetime(2000, 1, 6)],
                    [datetime(2000, 1, 2)],
                    None,
                    [datetime(2000, 1, 1)],
                    [datetime(2000, 1, 2)],
                ],
                dtype=pl.List(pl.Date),
            ),
            pl.Series(
                "str_key",
                [
                    ["z"],
                    ["z"],
                    ["x", "y"],
                    None,
                    None,
                    ["u", "x", "y"],
                    ["y"],
                    None,
                    ["z"],
                    ["y"],
                ],
                dtype=pl.List(pl.String),
            ),
            pl.Series(
                "B",
                [
                    [None],
                    [None],
                    [1, 2],
                    None,
                    None,
                    [1, 2, 6, 7, 8, 9, 10],
                    [1],
                    None,
                    [None],
                    [1],
                ],
                dtype=pl.List(pl.Int64),
            ),
        ]
    )

    assert result.equals(expected_result)


def test_str_key(df_map):
    df = pl.LazyFrame(
        {
            "mylist": [
                ["z"],
                ["y", "x"],
                [],
                None,
                ["y", "x", "u"],
                ["y", "y", "y"],
                ["_"],
                ["z", "_"],
                ["y", "_"],
            ]
        }
    )

    result = df.sf.join_on_list_items(
        df_map, left_on="mylist", right_on="str_key", how="left"
    ).collect()

    expected_result = pl.DataFrame(
        [
            pl.Series(
                "mylist",
                [
                    ["z"],
                    ["y", "x"],
                    [],
                    None,
                    ["y", "x", "u"],
                    ["y", "y", "y"],
                    ["_"],
                    ["z", "_"],
                    ["y", "_"],
                ],
                dtype=pl.List(pl.String),
            ),
            pl.Series(
                "date_key",
                [
                    [datetime(2000, 1, 1)],
                    [datetime(2000, 1, 2), datetime(2000, 1, 3)],
                    None,
                    None,
                    [datetime(2000, 1, 2), datetime(2000, 1, 3), datetime(2000, 1, 6)],
                    [datetime(2000, 1, 2)],
                    None,
                    [datetime(2000, 1, 1)],
                    [datetime(2000, 1, 2)],
                ],
                dtype=pl.List(pl.Date),
            ),
            pl.Series(
                "int_key",
                [[1], [2, 3], None, None, [2, 3, 6], [2], None, [1], [2]],
                dtype=pl.List(pl.Int64),
            ),
            pl.Series(
                "B",
                [
                    [None],
                    [1, 2],
                    None,
                    None,
                    [1, 2, 6, 7, 8, 9, 10],
                    [1],
                    None,
                    [None],
                    [1],
                ],
                dtype=pl.List(pl.Int64),
            ),
        ]
    )
    assert result.equals(expected_result)


def test_date_key(df_map):
    df = pl.LazyFrame(
        {
            "mylist": [
                [datetime(2000, 1, 1).date()],
                [datetime(2000, 1, 2).date(), datetime(2000, 1, 3).date()],
                [],
                None,
                [
                    datetime(2000, 1, 2).date(),
                    datetime(2000, 1, 3).date(),
                    datetime(2000, 1, 6).date(),
                ],
                [
                    datetime(2000, 1, 2).date(),
                    datetime(2000, 1, 2).date(),
                    datetime(2000, 1, 2).date(),
                ],
                [datetime(2000, 1, 9).date()],
                [datetime(2000, 1, 1).date(), datetime(2000, 1, 9).date()],
                [datetime(2000, 1, 2).date(), datetime(2000, 1, 9).date()],
            ]
        }
    )

    result = df.sf.join_on_list_items(
        df_map, left_on="mylist", right_on="date_key", how="left"
    ).collect()

    expected_result = pl.DataFrame(
        [
            pl.Series(
                "mylist",
                [
                    [datetime(2000, 1, 1).date()],
                    [datetime(2000, 1, 2).date(), datetime(2000, 1, 3).date()],
                    [],
                    None,
                    [
                        datetime(2000, 1, 2).date(),
                        datetime(2000, 1, 3).date(),
                        datetime(2000, 1, 6).date(),
                    ],
                    [
                        datetime(2000, 1, 2).date(),
                        datetime(2000, 1, 2).date(),
                        datetime(2000, 1, 2).date(),
                    ],
                    [datetime(2000, 1, 9).date()],
                    [datetime(2000, 1, 1).date(), datetime(2000, 1, 9).date()],
                    [datetime(2000, 1, 2).date(), datetime(2000, 1, 9).date()],
                ],
                dtype=pl.List(pl.Date),
            ),
            pl.Series(
                "int_key",
                [[1], [2, 3], None, None, [2, 3, 6], [2], None, [1], [2]],
                dtype=pl.List(pl.Int64),
            ),
            pl.Series(
                "str_key",
                [
                    ["z"],
                    ["x", "y"],
                    None,
                    None,
                    ["u", "x", "y"],
                    ["y"],
                    None,
                    ["z"],
                    ["y"],
                ],
                dtype=pl.List(pl.String),
            ),
            pl.Series(
                "B",
                [
                    [None],
                    [1, 2],
                    None,
                    None,
                    [1, 2, 6, 7, 8, 9, 10],
                    [1],
                    None,
                    [None],
                    [1],
                ],
                dtype=pl.List(pl.Int64),
            ),
        ]
    )

    assert result.equals(expected_result)

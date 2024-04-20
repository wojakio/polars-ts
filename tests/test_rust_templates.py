import polars as pl
from polars_ts import pig_latinnify, template_1


def test_piglatinnify():
    df = pl.from_dict(
        {
            "english": ["this", "is", "not", "pig", "latin"],
        }
    )
    result = df.with_columns(pig_latin=pig_latinnify("english"))

    expected_df = pl.from_dict(
        {
            "english": ["this", "is", "not", "pig", "latin"],
            "pig_latin": ["histay", "siay", "otnay", "igpay", "atinlay"],
        }
    )

    assert result.equals(expected_df)


def test_template_1():
    df = pl.from_dict({"x": [0, 1, 2, 3, 4]})

    result = df.lazy().with_columns(result=template_1(pl.col("x"), seed=100)).collect()

    expected_df = pl.from_dict(
        {
            "x": [0, 1, 2, 3, 4],
            "result": [100, 201, 303, 406, 510],
        }
    )

    assert result.equals(expected_df)

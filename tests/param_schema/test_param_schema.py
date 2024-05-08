import polars as pl
import pytest

from polars_ts.param_schema import ParamSchema


def test_basic():
    p1 = ParamSchema(
        [
            ("test", "a", pl.Float64),
            ("test", "b", pl.Float64, 0.0),
            ("test", "c", pl.Categorical, 1),
        ]
    )

    assert str(p1) == "ParamSchema<test=(a:Float64)[b:Float64=0.0,c:Categorical=1]>"

    with pytest.raises(ValueError, match="duplicate param spec"):
        ParamSchema(
            [
                ("test", "a", pl.Float64),
                ("test", "a", pl.Float64, 0.0),
                ("test", "c", pl.Categorical, 1),
            ]
        )

import datetime
import polars as pl
from polars.testing import assert_frame_equal
import polars_ts as ts # noqa
import pytest


@pytest.fixture
def df() -> pl.LazyFrame:
    t = datetime.datetime(2024, 1, 1, 0).date()
    nrows = 16
    result = pl.LazyFrame(
        [
            pl.Series("time", [
                t + datetime.timedelta(days=days + 1) for days in range(0, nrows * 2, 2)
            ]),
            pl.Series("item", ["A"] * (nrows//2) + ["B"] * (nrows//2), dtype=pl.Categorical),
            pl.Series("val1", list(range(0, nrows)), dtype=pl.Float64),
            pl.Series("val2", list(range(0, nrows)), dtype=pl.Float64) * 100.,
        ]
    )

    return result

@pytest.fixture
def params() -> pl.LazyFrame:
    result = (
        pl.LazyFrame([
            pl.Series("test_case", ["1", "2", "3", "4", "5"]),
            pl.Series("item", ["A", "A", "B", "B", "B"]),
            pl.Series("window_size", [1,10,1,10,50]),
            pl.Series("adjust",[True, True, False, False, False]),
        ])
        .with_columns(
            alpha=1/pl.col("window_size"),
        )
    )
    
    return result

def test_basic(df, params):
    result = df.mathx.ewm_mean_config(params).collect()
    # print(result.to_init_repr())

    expected_result = pl.DataFrame(
        [
            pl.Series("item", ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'], dtype=pl.Categorical(ordering='physical')),
            pl.Series("val2", [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 36.90036900369004, 100.0, 55.248618784530386, 2.0, 200.0, 90.59607823984761, 200.0, 113.94509549491097, 5.96, 300.0, 149.60808168477433, 300.0, 176.01427303246226, 11.8408, 400.0, 212.57862475259964, 400.0, 241.35417692013053, 19.603983999999997, 500.0, 279.0472426547616, 500.0, 309.83902807213724, 29.211904319999995, 600.0, 348.74433136233694, 600.0, 381.322808922536, 40.627666233599996, 700.0, 421.45297292864393, 700.0, 455.64319231401635, 53.81511290892799, 800.0, 800.0, 800.0, 800.0, 800.0, 900.0, 836.90036900369, 900.0, 855.2486187845303, 802.0, 1000.0, 890.5960782398475, 1000.0, 913.9450954949108, 805.96, 1100.0, 949.6080816847742, 1100.0, 976.0142730324621, 811.8408000000001, 1200.0, 1012.5786247525995, 1200.0, 1041.3541769201304, 819.6039840000001, 1300.0, 1079.0472426547615, 1300.0, 1109.8390280721371, 829.21190432, 1400.0, 1148.7443313623369, 1400.0, 1181.322808922536, 840.6276662336, 1500.0, 1221.452972928644, 1500.0, 1255.6431923140165, 853.8151129089281], dtype=pl.Float64),
            pl.Series("val1", [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.36900369003690037, 1.0, 0.5524861878453038, 0.02, 2.0, 0.9059607823984761, 2.0, 1.1394509549491099, 0.0596, 3.0, 1.4960808168477435, 3.0, 1.7601427303246227, 0.118408, 4.0, 2.1257862475259968, 4.0, 2.4135417692013057, 0.19603984, 5.0, 2.790472426547616, 5.0, 3.0983902807213726, 0.2921190432, 6.0, 3.48744331362337, 6.0, 3.8132280892253605, 0.40627666233599996, 7.0, 4.21452972928644, 7.0, 4.556431923140163, 0.53815112908928, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 8.3690036900369, 9.0, 8.552486187845304, 8.02, 10.0, 8.905960782398475, 10.0, 9.13945095494911, 8.0596, 11.0, 9.496080816847744, 11.0, 9.760142730324624, 8.118408, 12.0, 10.125786247525998, 12.0, 10.413541769201306, 8.196039840000001, 13.0, 10.790472426547618, 13.0, 11.098390280721373, 8.292119043200001, 14.0, 11.487443313623372, 14.0, 11.813228089225362, 8.406276662336001, 15.0, 12.21452972928644, 15.0, 12.556431923140163, 8.538151129089282], dtype=pl.Float64),
            pl.Series("time", [datetime.date(2024, 1, 2), datetime.date(2024, 1, 2), datetime.date(2024, 1, 2), datetime.date(2024, 1, 2), datetime.date(2024, 1, 2), datetime.date(2024, 1, 4), datetime.date(2024, 1, 4), datetime.date(2024, 1, 4), datetime.date(2024, 1, 4), datetime.date(2024, 1, 4), datetime.date(2024, 1, 6), datetime.date(2024, 1, 6), datetime.date(2024, 1, 6), datetime.date(2024, 1, 6), datetime.date(2024, 1, 6), datetime.date(2024, 1, 8), datetime.date(2024, 1, 8), datetime.date(2024, 1, 8), datetime.date(2024, 1, 8), datetime.date(2024, 1, 8), datetime.date(2024, 1, 10), datetime.date(2024, 1, 10), datetime.date(2024, 1, 10), datetime.date(2024, 1, 10), datetime.date(2024, 1, 10), datetime.date(2024, 1, 12), datetime.date(2024, 1, 12), datetime.date(2024, 1, 12), datetime.date(2024, 1, 12), datetime.date(2024, 1, 12), datetime.date(2024, 1, 14), datetime.date(2024, 1, 14), datetime.date(2024, 1, 14), datetime.date(2024, 1, 14), datetime.date(2024, 1, 14), datetime.date(2024, 1, 16), datetime.date(2024, 1, 16), datetime.date(2024, 1, 16), datetime.date(2024, 1, 16), datetime.date(2024, 1, 16), datetime.date(2024, 1, 18), datetime.date(2024, 1, 18), datetime.date(2024, 1, 18), datetime.date(2024, 1, 18), datetime.date(2024, 1, 18), datetime.date(2024, 1, 20), datetime.date(2024, 1, 20), datetime.date(2024, 1, 20), datetime.date(2024, 1, 20), datetime.date(2024, 1, 20), datetime.date(2024, 1, 22), datetime.date(2024, 1, 22), datetime.date(2024, 1, 22), datetime.date(2024, 1, 22), datetime.date(2024, 1, 22), datetime.date(2024, 1, 24), datetime.date(2024, 1, 24), datetime.date(2024, 1, 24), datetime.date(2024, 1, 24), datetime.date(2024, 1, 24), datetime.date(2024, 1, 26), datetime.date(2024, 1, 26), datetime.date(2024, 1, 26), datetime.date(2024, 1, 26), datetime.date(2024, 1, 26), datetime.date(2024, 1, 28), datetime.date(2024, 1, 28), datetime.date(2024, 1, 28), datetime.date(2024, 1, 28), datetime.date(2024, 1, 28), datetime.date(2024, 1, 30), datetime.date(2024, 1, 30), datetime.date(2024, 1, 30), datetime.date(2024, 1, 30), datetime.date(2024, 1, 30), datetime.date(2024, 2, 1), datetime.date(2024, 2, 1), datetime.date(2024, 2, 1), datetime.date(2024, 2, 1), datetime.date(2024, 2, 1)], dtype=pl.Date),
        ]
    ).with_columns(pl.col("time").cast(pl.Date)).select(result.columns)


    assert_frame_equal(result, expected_result)

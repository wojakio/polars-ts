from typing import Dict, Union

import polars as pl

from ..types import FrameType


def impl_date_to_imm_contract(
    df: FrameType,
    col_name: pl.Expr,
    prefix: pl.Expr,
    suffix: pl.Expr,
    out: str,
) -> FrameType:
    imm_dict = impl_month_to_imm_dict()

    result = df.with_columns(
        pl.when(col_name.is_null())
        .then(pl.lit(None).cast(pl.String))
        .otherwise(
            pl.concat_str(
                [
                    prefix,
                    pl.struct(
                        col_name.dt.month().alias("month"),
                        col_name.dt.strftime("%y").alias("year"),
                    ).map_elements(
                        lambda se: f"""{imm_dict.get(se['month'], "@@")}{se['year']}""",
                        return_dtype=pl.String,
                    ),
                    suffix,
                ]
            )
        )
        .cast(pl.Categorical)
        .alias(out)
    )

    return result


def impl_imm_contract_to_date(
    df: FrameType, col_name: str, default_day: int
) -> FrameType:
    imm_dict = impl_month_to_imm_dict(invert=True)
    result = df.with_columns(
        pl.col(col_name)
        .str.extract_groups(".*(?<imm>[F-Z])(?<year>[0-9]{2})")
        .map_elements(
            lambda se: (
                f"{str(default_day).rjust(2, '0')}-{str(imm_dict[se['imm']]).rjust(2, '0')}-{se['year']}"
            ),
            return_dtype=pl.String,
        )
        .str.to_date("%d-%m-%y")
    )

    return result


def impl_int_to_date(df: FrameType, col_name: str) -> FrameType:
    result = df.with_columns(
        pl.col(col_name).cast(pl.String).str.strptime(pl.Date, "%Y%m%d")
    )

    return result


def impl_date_to_int(df: FrameType, col_name: str, out: str) -> FrameType:
    result = df.with_columns(
        (
            (pl.col(col_name).dt.year() * 10000)
            + (pl.col(col_name).dt.month().cast(pl.UInt16) * 100)
            + (pl.col(col_name).dt.day())
        )
        .alias(out)
        .cast(pl.UInt64)
    )

    return result


def impl_month_to_imm_dict(
    invert: bool = False,
) -> Dict[Union[str, int], Union[str, int]]:
    result_cols = ["month_idx", "imm_code"]
    df = (
        pl.from_dict({"imm_code": "FGHJKMNQUVXZ"})
        .select(
            pl.col("imm_code").str.extract_all("[F-Z]").explode().cast(pl.Categorical)
        )
        .with_row_index("month_idx", offset=1)
        .select(reversed(result_cols) if invert else result_cols)
    )

    return dict(df.iter_rows())

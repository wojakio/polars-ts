from typing import Dict, Union

import polars as pl


def month_to_imm(
    as_dict: bool = False, invert: bool = False
) -> Dict[Union[str, int], Union[str, int]]:
    result_cols = ["month_idx", "imm_code"]
    df = (
        pl.DataFrame({"imm_code": "FGHJKMNQUVXZ"})
        .select(
            pl.col("imm_code").str.extract_all("[F-Z]").explode().cast(pl.Categorical)
        )
        .with_row_index("month_idx", offset=1)
        .select(reversed(result_cols) if invert else result_cols)
    )

    return dict(df.iter_rows()) if as_dict else df

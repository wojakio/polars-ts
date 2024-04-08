import hashlib

import polars as pl


def _get_unique_col_name(df: pl.DataFrame, prefix: str) -> str:
    i = 0
    unique_col_name = f"{prefix}{i}"
    while unique_col_name in df.columns:
        i += 1
        unique_col_name = f"{prefix}{i}"

    return unique_col_name


def _deterministic_seed(name: str, seed_multiplier: int) -> int:
    seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (10**8)
    seed = (seed * seed_multiplier) % 2**31
    return int(seed)

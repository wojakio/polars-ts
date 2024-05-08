from typing import Tuple
import polars as pl

from ..types import FrameType
from ..sf_helper import column_name_unique_over


def impl_construct_closure(
    conv_df: FrameType,
    max_iterations: int,
) -> FrameType:
    # close under identity, inverse and transitive
    cf = pl.col("conversion_factor")

    inverses_df = conv_df.with_columns(
        base="target", target="base", conversion_factor=1 / cf
    )

    known_units = pl.concat(
        [conv_df.select("base"), conv_df.select(base="target")]
    ).unique()

    # unusual construction to align schema of identities_df with conv_df
    identities_df = (
        conv_df.clear()
        .join(known_units, on="base", how="outer_coalesce")
        .with_columns(
            target="base",
            conversion_factor=1.0,
        )
    )

    partial = (
        pl.concat([conv_df, inverses_df, identities_df])
        .with_columns(
            pl.exclude("base", "target", "conversion_factor").forward_fill(),
        )
        .lazy()
        .collect()
    )

    added_rows = len(partial)
    while (added_rows > 0) and (max_iterations != 0):
        old_rows = len(partial)
        max_iterations = max_iterations - 1

        partial = (
            partial.join(partial, left_on="target", right_on="base", how="inner")
            .select(
                "base",
                target="target_right",
                conversion_factor=cf * pl.col("conversion_factor_right"),
            )
            .unique(subset=["base", "target"], keep="first")
            .sort("base", "target")
        )

        added_rows = len(partial) - old_rows

    result: FrameType = (
        partial.lazy()
        if isinstance(conv_df, pl.LazyFrame)
        else partial.lazy().collect()
    )

    return result


def _split_multi_unit(u: pl.Expr, field_name: str) -> Tuple[pl.Expr, str, str]:
    num = f"{field_name}_numerator"
    den = f"{field_name}_denominator"

    split_unit_expr = (
        u.cast(pl.String)
        .str.split_exact("/", 1)
        .struct.rename_fields([num, den])
        .cast(pl.Struct({num: pl.Categorical, den: pl.Categorical}))
        .alias(field_name)
    )

    return split_unit_expr, num, den


def _impl_convert_multi_dim(
    df: FrameType,
    target_unit: pl.Expr,
    conversion_matrix: FrameType,
    value: pl.Expr,
    value_unit: pl.Expr,
) -> FrameType:
    value_unit_name = value_unit.meta.output_name()
    target_unit_name = target_unit.meta.output_name()

    value_unit_prefix = column_name_unique_over(value_unit_name, df)
    target_unit_prefix = column_name_unique_over(target_unit_name, df)

    value_unit_list, value_num, value_den = _split_multi_unit(
        value_unit, value_unit_prefix
    )
    target_unit_list, target_num, target_den = _split_multi_unit(
        target_unit, target_unit_prefix
    )

    result = (
        df.with_columns(
            target_unit_list,
            value_unit_list,
        )
        .unnest(target_unit_prefix, value_unit_prefix)
        .pipe(
            _impl_convert_single_dim,
            pl.col(target_num),
            conversion_matrix,
            value,
            value_unit=pl.col(value_num),
            invert=False,
        )
        .pipe(
            _impl_convert_single_dim,
            pl.col(target_den),
            conversion_matrix,
            value,
            value_unit=pl.col(value_den),
            invert=True,
        )
        .with_columns(
            pl.concat_str(
                [value_num, value_den],
                separator="/",
            ).alias(value_unit_name)
        )
        .select(pl.exclude(f"^{target_unit_prefix}.*$", f"^{value_unit_prefix}.*$"))
    )

    return result


def _impl_convert_single_dim(
    df: FrameType,
    target_unit: pl.Expr,
    conversion_matrix: FrameType,
    value: pl.Expr,
    value_unit: pl.Expr,
    invert: bool,
) -> FrameType:
    value_unit_name = value_unit.meta.output_name()
    target_unit_name = target_unit.meta.output_name()

    conversion_join_cols = ["target", "base"] if invert else ["base", "target"]
    working_columns = []

    if target_unit_name not in df.columns:
        df = df.with_columns(target_unit.alias(target_unit_name))
        working_columns.append(target_unit_name)

    result = (
        df.join(
            conversion_matrix,
            left_on=[value_unit_name, target_unit_name],
            right_on=conversion_join_cols,
            how="left",
        )
        .with_columns(
            (value * pl.coalesce("conversion_factor", 1.0)).alias(
                value.meta.output_name()
            ),
            pl.when(pl.col("conversion_factor").is_null())
            .then(value_unit_name)
            .otherwise(target_unit_name),
        )
        .select(pl.exclude("conversion_factor", *working_columns))
    )

    return result


def impl_convert(
    df: FrameType,
    target_unit: pl.Expr,
    conversion_matrix: FrameType,
    value: pl.Expr,
    value_unit: pl.Expr,
    is_multi_dim: bool,
) -> FrameType:
    if is_multi_dim:
        result = _impl_convert_multi_dim(
            df, target_unit, conversion_matrix, value, value_unit
        )
    else:
        result = _impl_convert_single_dim(
            df, target_unit, conversion_matrix, value, value_unit, invert=False
        )

    return result

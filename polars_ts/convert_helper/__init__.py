import polars as pl

from ..types import FrameType

def impl_construct_closure(
    conv_df: FrameType,
    max_iterations: int,
) -> FrameType:

    # construct identity, inverse and transitive
    value = pl.col("value")

    inverses_df = (
        conv_df
        .with_columns(base="target", target="base", value=1/value)
    )

    known_units = pl.concat([conv_df.select("base"), conv_df.select(base="target")]).unique()

    # unusual construction to align schema of identities_df with conv_df
    identities_df = (
        conv_df
        .clear()
        .join(known_units, on="base", how="outer_coalesce")
        .with_columns(target="base", value=1.0,)
    )

    partial = (
        pl.concat([conv_df, inverses_df, identities_df])
        .with_columns(
            pl.col("base").cast(pl.String).str.to_lowercase(),
            pl.col("target").cast(pl.String).str.to_lowercase(),
            pl.exclude("base", "target", "value").forward_fill(),
        )
        .collect()
    )

    added_rows = len(partial)
    while (added_rows > 0) and (max_iterations != 0) :
        old_rows = len(partial)
        max_iterations = max_iterations - 1

        partial = (
            partial
            .join(partial, left_on="target", right_on="base", how="inner")
            .select(
                "base",
                target="target_right",
                value=value * pl.col("value_right")
            )
            .unique(subset=["base", "target"], keep="first")
            .sort("base", "target")
        )

        added_rows = len(partial) - old_rows

    result = partial.with_columns(pl.col(pl.String).cast(pl.Categorical))

    if isinstance(conv_df, pl.LazyFrame):
        result = result.lazy()

    return result



def impl_convert(
    df: FrameType,
    target_unit: pl.Expr,
    conversion_matrix: FrameType,
    value: pl.Expr,
    value_unit: pl.Expr,
    strict: bool,
) -> FrameType:
    
    is_target_multi_unit = target_unit.cast(pl.String).str.contains('/').alias("is_target_multi_unit")
    need_invert = value_unit.cast(pl.String).str.starts_with("~").alias("need_invert")

    conversions = (
        conversion_matrix
        .filter(target=target_unit.cast(pl.String).str.to_lowercase())
        .rename({value.meta.output_name(): "conversion_factor"})
    )

    if conversions.lazy().collect().is_empty():
        raise ValueError(f"Unknown target_unit: {str(target_unit)}")

    # df = df.lazy().collect()
    # conversions = conversions.lazy().collect()

    result = (
        df
        .with_columns(is_target_multi_unit, need_invert)
    )

    result = (
        result
        .with_columns(
            pl.when(pl.col("need_invert"))
              .then(1/value)
              .otherwise(value)
              .alias(value.meta.output_name()),
            pl.when(pl.col("need_invert"))
              .then(value_unit.cast(pl.String).str.strip_prefix("~"))
              .otherwise(value_unit)
              .alias(value_unit.meta.output_name())
        )
        .join(
            conversions,
            left_on=value_unit.cast(pl.String).str.to_lowercase().cast(pl.Categorical),
            right_on="base",
            how="left"
        )
        .with_columns(
            (value * pl.coalesce("conversion_factor", 1.0)).alias(value.meta.output_name()),
            pl.coalesce("target", value_unit).alias(value_unit.meta.output_name())
        )
        .with_columns(
            pl.when(pl.col("need_invert"))
              .then(1/value)
              .otherwise(value)
              .alias(value.meta.output_name()),
            pl.when(pl.col("need_invert"))
              .then(pl.concat_str([pl.lit("~"), value_unit]).cast(pl.Categorical))
              .otherwise(value_unit)
              .alias(value_unit.meta.output_name())
        )
        .drop("target", "conversion_factor", "need_invert", "is_target_multi_unit")
    )

    return result

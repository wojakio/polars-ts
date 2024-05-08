// copy of: polars-plan/src/dsl/function_expr/range/utils.rs
use polars_core::prelude::{
    polars_bail, ChunkedArray, IntoSeries, ListBuilderTrait, ListChunked,
    ListPrimitiveChunkedBuilder, PolarsIntegerType, PolarsResult, Series,
};

use polars_arrow::array::Array;
use polars_arrow::types::NativeType;

use polars_core::utils::arrow::array::PrimitiveArray;

fn collect_into_vec_remove_nulls<T>(arr: &(dyn Array + 'static)) -> Vec<T>
where
    T: NativeType,
{
    if arr.is_empty() {
        let empty: Vec<T> = Vec::new();
        return empty;
    }

    arr.as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap()
        .into_iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>()
        .clone()
}

/// Create a ranges column from the given start/end columns and a range function.
pub(crate) fn temporal_ranges_impl_broadcast<T, U, F>(
    start: &ChunkedArray<T>,
    end: &ChunkedArray<T>,
    skips: &ListChunked,
    weekends: &ListChunked,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<Series>
where
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(
        T::Native,
        T::Native,
        Vec<i32>,
        Vec<i64>,
        &mut ListPrimitiveChunkedBuilder<U>,
    ) -> PolarsResult<()>,
{
    match (start.len(), end.len()) {
        (len_start, len_end) if len_start == len_end => {
            build_temporal_ranges::<_, _, T, U, F>(
                start.downcast_iter().flatten(),
                end.downcast_iter().flatten(),
                skips.downcast_iter().flatten(),
                weekends.downcast_iter().flatten(),
                range_impl,
                builder,
            )?;
        }
        (1, len_end) => {
            let start_scalar = start.get(0);
            match start_scalar {
                Some(start) => build_temporal_ranges::<_, _, T, U, F>(
                    std::iter::repeat(Some(&start)),
                    end.downcast_iter().flatten(),
                    skips.downcast_iter().flatten(),
                    weekends.downcast_iter().flatten(),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len_end),
            }
        }
        (len_start, 1) => {
            let end_scalar = end.get(0);
            match end_scalar {
                Some(end) => build_temporal_ranges::<_, _, T, U, F>(
                    start.downcast_iter().flatten(),
                    std::iter::repeat(Some(&end)),
                    skips.downcast_iter().flatten(),
                    weekends.downcast_iter().flatten(),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len_start),
            }
        }
        (len_start, len_end) => {
            polars_bail!(
                ComputeError:
                "lengths of `start` ({}) and `end` ({}) do not match",
                len_start, len_end
            )
        }
    };
    let out = builder.finish().into_series();
    Ok(out)
}

/// Iterate over a start and end column and create a range for each entry.
fn build_temporal_ranges<'a, I, J, T, U, F>(
    start: I,
    end: J,
    skips: impl Iterator<Item = Option<Box<dyn Array>>>,
    weekends: impl Iterator<Item = Option<Box<dyn Array>>>,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<()>
where
    I: Iterator<Item = Option<&'a T::Native>>,
    J: Iterator<Item = Option<&'a T::Native>>,
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(
        T::Native,
        T::Native,
        Vec<i32>,
        Vec<i64>,
        &mut ListPrimitiveChunkedBuilder<U>,
    ) -> PolarsResult<()>,
{
    for (start, end, skips, weekends) in start
        .zip(end)
        .zip(skips)
        .zip(weekends)
        .map(|(((a, b), c), d)| (a, b, c, d))
    {
        match (start, end, skips, weekends) {
            (Some(start), Some(end), Some(skips), Some(weekends)) => {
                let skip = collect_into_vec_remove_nulls::<i32>(&*skips);
                let weekend = collect_into_vec_remove_nulls::<i64>(&*weekends);
                range_impl(*start, *end, skip, weekend, builder)?
            }
            _ => builder.append_null(),
        }
    }
    Ok(())
}

/// Add `n` nulls to the builder.
fn build_nulls<U>(builder: &mut ListPrimitiveChunkedBuilder<U>, n: usize)
where
    U: PolarsIntegerType,
{
    for _ in 0..n {
        builder.append_null()
    }
}

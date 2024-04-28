#![allow(clippy::unused_unit)]
use crate::math::{impl_random_normal, impl_random_uniform, impl_wyhash};
// use crate::time::impl_datetime_ranges_custom;
use crate::time::utils::temporal_ranges_impl_broadcast;
use crate::utils::same_output_type;
use polars_ops::series::{diff, ewm_mean, pct_change, EWMOptions};
use polars_time::chunkedarray::DateMethods;
use polars_time::{datetime_range_impl, ClosedWindow, Duration};

use polars_arrow::temporal_conversions::MILLISECONDS_IN_DAY;
use std::collections::HashSet;
use std::ops::Add;

use polars::prelude::*;

use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::CustomIterTools;

use serde::Deserialize;
use std::fmt::Write;

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(|value: &str, output: &mut String| {
        if let Some(first_char) = value.chars().next() {
            write!(output, "{}{}ay", &value[1..], first_char).unwrap()
        }
    });
    Ok(out.into_series())
}

#[derive(Deserialize)]
struct PlTemplate1Kwargs {
    seed: i64,
}

#[polars_expr(output_type_func=same_output_type)]
fn pl_template_1(inputs: &[Series], kwargs: PlTemplate1Kwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca: &Int64Chunked = s.i64()?;
    let out: Int64Chunked = ca
        .into_iter()
        .scan(0_i64, |state: &mut i64, x: Option<i64>| match x {
            Some(x) => {
                *state += x + kwargs.seed;
                Some(Some(*state))
            }
            None => Some(None),
        })
        .collect_trusted();
    let out: Int64Chunked = out.with_name(ca.name());
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn pl_random_uniform(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &ChunkedArray<UInt32Type> = inputs[0].u32()?;
    let len = ca.get(0).unwrap() as usize;
    let low = inputs[1].f64()?;
    let low = low.get(0).unwrap();
    let high = inputs[2].f64()?;
    let high = high.get(0).unwrap();
    let seed = inputs[3].u64()?;
    let seed = seed.get(0);

    let result = impl_random_uniform(len, low, high, seed)?
        .with_name(ca.name())
        .into_series();
    Ok(result)
}

#[polars_expr(output_type=Float64)]
fn pl_random_normal(inputs: &[Series]) -> PolarsResult<Series> {
    let len = inputs[0].u32()?;
    let len = len.get(0).unwrap() as usize;
    let mean = inputs[1].f64()?;
    let mean = mean.get(0).unwrap();
    let std_ = inputs[2].f64()?;
    let std_ = std_.get(0).unwrap();
    let seed = inputs[3].u64()?;
    let seed = seed.get(0);

    let result = impl_random_normal(len, mean, std_, seed)?
        // .with_name(ca.name())
        .into_series();
    Ok(result)
}

#[polars_expr(output_type=UInt64)]
fn pl_wyhash(inputs: &[Series]) -> PolarsResult<Series> {
    let s = inputs.first().unwrap();
    let result = impl_wyhash(s)?.into_series();
    Ok(result)
}

fn list_i64_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let dtype = DataType::List(Box::new(DataType::Int64));
    let field = Field::new(input_fields[0].name(), dtype);
    Ok(field.clone())
}

const CAPACITY_FACTOR: usize = 5;

#[polars_expr(output_type_func=list_i64_dtype)]
fn pl_datetime_ranges_custom(inputs: &[Series]) -> PolarsResult<Series> {
    // thread::sleep(std::time::Duration::from_millis(4000));

    let start = &inputs[0];
    let end = &inputs[1];

    let start = start.cast(&DataType::Int64)?;
    let end = end.cast(&DataType::Int64)?;

    let start: ChunkedArray<Int64Type> = start.i64().unwrap() * MILLISECONDS_IN_DAY;
    let end: ChunkedArray<Int64Type> = end.i64().unwrap() * MILLISECONDS_IN_DAY;

    let empty32 = ListChunked::full_null_with_dtype("skips", start.len(), &DataType::Int32);
    let empty64 = ListChunked::full_null_with_dtype("weekend", start.len(), &DataType::Int64);

    let skips: &ListChunked = inputs[2].list().unwrap_or(&empty32); //.unwrap_or(ChunkedArray<ListType>::from_);
    let weekends: &ListChunked = inputs[3].list().unwrap_or(&empty64);

    // builder for the output Series = ListChunked = ChunkedArray< ListType > = ChunkedArray< List<Date> >
    let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
        start.name(),
        start.len(),
        start.len() * CAPACITY_FACTOR,
        DataType::Int32,
    );

    let interval = Duration::parse("1d");
    let closed = ClosedWindow::Both;

    let range_per_row_impl =
        |start,
         end,
         skips: Vec<i32>,
         weekends: Vec<i64>,
         builder: &mut ListPrimitiveChunkedBuilder<Int32Type>| {
            let rng = datetime_range_impl(
                "",
                start,
                end,
                interval,
                closed,
                TimeUnit::Milliseconds,
                None,
            )?;
            let rng = rng.cast(&DataType::Date).unwrap();

            let skips: Vec<i64> = skips
                .iter()
                .map(|&v| (v as i64) * MILLISECONDS_IN_DAY)
                .collect();
            let skips_set = skips.into_iter().collect::<HashSet<_>>();

            let rng_i64 = rng.cast(&DataType::Int64)?;
            let rng_i64 = rng_i64.i64().unwrap() * MILLISECONDS_IN_DAY; // Ensure this is a date type, handle errors as needed
            let holiday_mask: BooleanChunked =
                rng_i64.apply_values_generic(|dt| !skips_set.contains(&dt));

            let rng_no_hols = rng.filter(&holiday_mask)?;

            let rng_weekends: &ChunkedArray<Int8Type> =
                &rng_no_hols.date().map(|ca| ca.weekday()).unwrap();

            let rng_weekends = rng_weekends.cast(&DataType::Int64).unwrap();
            let rng_weekends: &ChunkedArray<Int64Type> = rng_weekends.i64().unwrap();
            let weekends_mask: BooleanChunked =
                rng_weekends.apply_values_generic(|wd| !weekends.contains(&wd));

            let rng_no_hols_wd_only = rng_no_hols.filter(&weekends_mask)?;

            let rng = rng_no_hols_wd_only.to_physical_repr();
            let rng = rng.i32().unwrap();
            builder.append_iter(rng.iter());
            Ok(())
        };

    let out = temporal_ranges_impl_broadcast(
        &start,
        &end,
        skips,
        weekends,
        range_per_row_impl,
        &mut builder,
    )?;

    let to_type = DataType::List(Box::new(DataType::Date));
    let result = out.cast(&to_type)?.into_series();
    Ok(result)
}

#[polars_expr(output_type_func=same_output_type)]
fn pl_ewm_custom(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let alpha = inputs[1].f64()?.get(0).unwrap();
    let min_periods = inputs[2].u64()?.get(0).unwrap() as usize;
    let adjust = inputs[3].bool()?.get(0).unwrap();

    let options = EWMOptions {
        alpha,
        adjust,
        bias: false,
        min_periods,
        ignore_nulls: true,
    };

    // dbg!(&options);

    ewm_mean(s, options)
}

#[polars_expr(output_type_func=same_output_type)]
fn pl_shift_custom(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let n = inputs[1].i64()?.get(0).unwrap();

    let shifted = s.shift(n);
    Ok(shifted)
}

#[polars_expr(output_type_func=same_output_type)]
fn pl_diff_custom(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let n = inputs[1].i64()?.get(0).unwrap();
    let method = inputs[2].str()?.get(0).unwrap();

    match method {
        "arithmetic" => diff(s, n, polars::series::ops::NullBehavior::Ignore),
        "fractional" => {
            let n = Series::new("n__", &[n]);
            pct_change(s, &n)
        }
        "geometric" => {
            let n = Series::new("n__", &[n]);
            let res = pct_change(s, &n).unwrap().add(1_f64);
            Ok(res)
        }
        _ => {
            panic!("unknown diff method `{:?}`", method)
        }
    }
}

#[cfg(test)]
mod test {
    use polars::prelude::*;

    #[test]
    fn test_series_equals() {
        let a = Series::new("a", &[1_u32, 2, 3]);
        let b = Series::new("a", &[1_u32, 2, 3]);
        assert!(a.equals(&b));

        let s = Series::new("foo", &[None, Some(1i64)]);
        assert!(s.equals_missing(&s));
    }
}

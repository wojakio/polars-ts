#![allow(clippy::unused_unit)]
use crate::math::{impl_random_normal, impl_random_uniform, impl_wyhash};
use crate::utils::same_output_type;

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

#![allow(clippy::unused_unit)]
use polars::prelude::*;

use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Normal, StandardNormal};
use wyhash::wyhash;

pub(crate) fn impl_random_uniform(
    len: usize,
    low: f64,
    high: f64,
    seed: Option<u64>,
) -> Result<ChunkedArray<Float64Type>, PolarsError> {
    if low >= high {
        return Err(PolarsError::ComputeError(
            "Low must be strictly less than high for a uniform distribution".into(),
        ));
    }

    let dist = Uniform::new(low, high);
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };

    let out = Float64Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(len));

    Ok(out)
}

pub(crate) fn impl_random_normal(
    len: usize,
    mean: f64,
    std_dev: f64,
    seed: Option<u64>,
) -> Result<ChunkedArray<Float64Type>, PolarsError> {
    if std_dev <= 0.0 {
        return Err(PolarsError::ComputeError(
            "Standard deviation must be positive".into(),
        ));
    }

    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };

    let out = if mean == 0.0 && std_dev == 1.0 {
        Float64Chunked::from_iter_values("", (&mut rng).sample_iter(StandardNormal).take(len))
    } else {
        let dist = Normal::new(mean, std_dev).unwrap();
        Float64Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(len))
    };

    Ok(out)
}

pub(crate) fn impl_wyhash(s: &Series) -> Result<ChunkedArray<UInt64Type>, PolarsError> {
    let ca = s.str()?;

    let out: ChunkedArray<UInt64Type> = ca
        .apply_generic(|v| v.map(|v| wyhash(v.as_bytes(), 42)))
        .with_name(ca.name());

    Ok(out)
}

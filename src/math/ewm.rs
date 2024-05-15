use polars_arrow::array::PrimitiveArray;
use polars_arrow::legacy::kernels::ewm::ewm_mean as ewm_mean_polars;
use polars_arrow::trusted_len::TrustedLen;

use std::hash::{Hash, Hasher};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) enum OutlierOptions {
    None,
    Threshold { threshold: f64 },
    Winsor { lower: f64, upper: f64 },
    Trim { lower: f64, upper: f64 },
}

impl Hash for OutlierOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match *self {
            OutlierOptions::None => {
                0.hash(state);
            }
            OutlierOptions::Threshold { threshold } => {
                1.hash(state);
                threshold.to_bits().hash(state);
            }
            OutlierOptions::Winsor { lower, upper } => {
                2.hash(state);
                lower.to_bits().hash(state);
                upper.to_bits().hash(state);
            }
            OutlierOptions::Trim { lower, upper } => {
                3.hash(state);
                lower.to_bits().hash(state);
                upper.to_bits().hash(state);
            }
        }
    }
}

// polars-ops-0.39.2/src/series/ops/ewm.rs
pub(crate) fn impl_ewm_mean_f64<I>(
    xs: I,
    alpha: f64,
    adjust: bool,
    min_periods: usize,
    ignore_nulls: bool,
    outlier_strategy: OutlierOptions,
) -> PrimitiveArray<f64>
where
    I: IntoIterator<Item = Option<f64>>,
    I::IntoIter: TrustedLen,
{
    match outlier_strategy {
        OutlierOptions::None => ewm_mean_polars(xs, alpha, adjust, min_periods, ignore_nulls),
        OutlierOptions::Threshold { threshold } => {
            ewm_mean_threshold(xs, alpha, adjust, min_periods, ignore_nulls, threshold)
        }
        OutlierOptions::Trim {
            lower: _lower,
            upper: _upper,
        } => {
            panic!("Not implemented: ewm_mean(trim)")
        }
        OutlierOptions::Winsor {
            lower: _lower,
            upper: _upper,
        } => {
            panic!("Not implemented: ewm_mean(winsor)")
        }
    }
}

fn ewm_mean_threshold<I>(
    xs: I,
    alpha: f64,
    _adjust: bool,
    min_periods: usize,
    _ignore_nulls: bool,
    threshold: f64,
) -> PrimitiveArray<f64>
where
    I: IntoIterator<Item = Option<f64>>,
{
    let old_wt_factor = 1.0 - alpha;
    let mut previous_ewma: Option<f64> = None;
    let mut valid_count = 0;

    xs.into_iter()
        .enumerate()
        .map(|(index, x_opt)| {
            if let Some(x) = x_opt {
                valid_count += 1;
                let previous_weighted_average = previous_ewma.unwrap_or(x);
                let new_wt = old_wt_factor.powi(index as i32);

                let weight = 1.0 / (1.0 - new_wt * old_wt_factor);
                let lambda_term = previous_weighted_average * old_wt_factor * (1.0 - new_wt);
                let clipped_value =
                    (x.min(threshold * previous_weighted_average)) * (1.0 - old_wt_factor);

                previous_ewma = Some(weight * (lambda_term + clipped_value));
            } else {
                valid_count = valid_count.max(0usize);
            }

            if valid_count >= min_periods {
                previous_ewma
            } else {
                None
            }
        })
        .collect()
}

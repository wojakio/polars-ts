// #![allow(clippy::unused_unit)]
// use polars::prelude::*;

pub(crate) mod utils;

// const CAPACITY_FACTOR: usize = 5;

// pub(crate) fn impl_datetime_ranges_custom(
//     start: i64,
//     end: i64,
//     skips: &[i32],
//     weekends: &[i8],
// ) -> PolarsResult<Series> {

//     let mut builder =
//         ListPrimitiveChunkedBuilder::<Int64Type>::new(
//             "time",
//             start.len(),
//             start.len() * CAPACITY_FACTOR,
//             DataType.Int32
//     );

//     let range_impl = |start, end, builder: &mut ListPrimitiveChunkedBuilder<Int32Type>| {
//         let rng = date_ranges_custom_impl(
//             "",
//             start,
//             end,
//             skips,
//             weekends,
//             TimeUnit::Milliseconds,
//             None
//         )?;

//         let rng = rng.cast(&DataType::Date).unwrap();
//         let rng = rng.to_physical_repr();
//         let rng = rng.i32().unwrap();
//         builder.append_slice(rng.cont_slice().unwrap());
//         Ok(())
//     };

//     let out = utils::temporal_ranges_impl_broadcast(&start, &end, range_impl, &mut_builder)?;
//     Ok(out)

// }

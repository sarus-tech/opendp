use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Sum;
use std::ops::{Div, Mul, Sub, AddAssign};
use std::os::raw::{c_char, c_uint, c_void};
use std::str::FromStr;

use num::{One, Integer, Zero, NumCast};

use opendp::core::{DatasetMetric, Metric, SensitivityMetric};
use opendp::dist::{HammingDistance, L1Sensitivity, L2Sensitivity, SymmetricDistance};
use opendp::dom::{AllDomain, VectorDomain};
use opendp::traits::{Abs, CastFrom, DistanceCast};
use opendp::trans::{count, MakeTransformation0, MakeTransformation1, MakeTransformation2, MakeTransformation3, manipulation, sum};
use opendp::trans;
use opendp::trans::sum::{BoundedSum, BoundedSumConstant};

use crate::core::{FfiTransformation, FfiObject, FfiResult};
use crate::util;
use crate::util::{c_bool, Type, TypeArgs, TypeContents};
use opendp::trans::count::{CountByCategoriesConstant, CountByCategories, CountBy, CountByConstant};
use num::traits::FloatConst;

#[no_mangle]
pub extern "C" fn opendp_trans__make_identity(type_args: *const c_char) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize_scalar<T: 'static + Clone>() -> FfiResult<*mut FfiTransformation> {
        manipulation::Identity::make(AllDomain::<T>::new(), HammingDistance::new()).into()
    }
    fn monomorphize_vec<T: 'static + Clone>() -> FfiResult<*mut FfiTransformation> {
        manipulation::Identity::make(VectorDomain::new(AllDomain::<T>::new()), HammingDistance::new()).into()
    }
    let type_args = try_!(TypeArgs::parse(type_args, 1));
    match &type_args.0[0].contents {
        TypeContents::VEC(element_id) => dispatch!(monomorphize_vec, [(try_!(Type::of_id(*element_id)), @primitives)], ()),
        _ => dispatch!(monomorphize_scalar, [(&type_args.0[0], @primitives)], ())
    }
}

#[no_mangle]
pub extern "C" fn opendp_trans__make_split_lines(type_args: *const c_char) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<M>() -> FfiResult<*mut FfiTransformation>
        where M: 'static + DatasetMetric<Distance=u32> + Clone {
        trans::SplitLines::<M>::make().into()
    }
    let type_args = try_!(TypeArgs::parse(type_args, 1));
    dispatch!(monomorphize, [(type_args.0[0], @dist_dataset)], ())
}

#[no_mangle]
pub extern "C" fn opendp_trans__make_parse_series(type_args: *const c_char, impute: c_bool) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<M, T>(impute: bool) -> FfiResult<*mut FfiTransformation>
        where M: 'static + DatasetMetric<Distance=u32> + Clone,
              T: 'static + FromStr + Default,
              T::Err: Debug {
        trans::ParseSeries::<T, M>::make(impute).into()
    }
    let type_args = try_!(TypeArgs::parse(type_args, 1));
    let impute = util::to_bool(impute);
    dispatch!(monomorphize, [(type_args.0[0], @dist_dataset), (type_args.0[1], @primitives)], (impute))
}

#[no_mangle]
pub extern "C" fn opendp_trans__make_split_records(type_args: *const c_char, separator: *const c_char) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<M>(separator: Option<&str>) -> FfiResult<*mut FfiTransformation>
        where M: 'static + DatasetMetric<Distance=u32> + Clone {
        trans::SplitRecords::<M>::make(separator).into()
    }
    let type_args = try_!(TypeArgs::parse(type_args, 1));
    let separator = try_!(util::to_option_str(separator));
    dispatch!(monomorphize, [(type_args.0[0], @dist_dataset)], (separator))
}

#[no_mangle]
pub extern "C" fn opendp_trans__make_create_dataframe(type_args: *const c_char, col_names: *const FfiObject) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<M, K>(col_names: *const FfiObject) -> FfiResult<*mut FfiTransformation>
        where M: 'static + DatasetMetric<Distance=u32> + Clone,
              K: 'static + Eq + Hash + Debug + Clone {
        let col_names = try_as_ref!(col_names).as_ref::<Vec<K>>().clone();
        trans::CreateDataFrame::<M, K>::make(col_names).into()
    }
    let type_args = try_!(TypeArgs::parse(type_args, 2));
    dispatch!(monomorphize, [(type_args.0[0], @dist_dataset), (type_args.0[1], @hashable)], (col_names))
}

#[no_mangle]
pub extern "C" fn opendp_trans__make_split_dataframe(type_args: *const c_char, separator: *const c_char, col_names: *const FfiObject) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<M, K>(separator: Option<&str>, col_names: *const FfiObject) -> FfiResult<*mut FfiTransformation>
        where M: 'static + DatasetMetric<Distance=u32> + Clone,
              K: 'static + Eq + Hash + Debug + Clone {
        let col_names = try_as_ref!(col_names).as_ref::<Vec<K>>().clone();
        trans::SplitDataFrame::<M, K>::make(separator, col_names).into()
    }
    let type_args = try_!(TypeArgs::parse(type_args, 2));
    let separator = try_!(util::to_option_str(separator));
    dispatch!(monomorphize, [(type_args.0[0], @dist_dataset), (type_args.0[1], @hashable)], (separator, col_names))
}

#[no_mangle]
pub extern "C" fn opendp_trans__make_parse_column(type_args: *const c_char, key: *const c_void, impute: c_bool) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<M, K, T>(key: *const c_void, impute: bool) -> FfiResult<*mut FfiTransformation> where
        M: 'static + DatasetMetric<Distance=u32> + Clone,
        K: 'static + Hash + Eq + Debug + Clone,
        T: 'static + Debug + Clone + PartialEq + FromStr + Default,
        T::Err: Debug {
        let key = try_as_ref!(key as *const K).clone();
        trans::ParseColumn::<M, T>::make(key, impute).into()
    }
    let type_args = try_!(TypeArgs::parse(type_args, 3));
    let impute = util::to_bool(impute);
    dispatch!(monomorphize, [(type_args.0[0], @dist_dataset), (type_args.0[1], @hashable), (type_args.0[2], @primitives)], (key, impute))
}

#[no_mangle]
pub extern "C" fn opendp_trans__make_select_column(type_args: *const c_char, key: *const c_void) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<M, K, T>(key: *const c_void) -> FfiResult<*mut FfiTransformation> where
        M: 'static + DatasetMetric<Distance=u32> + Clone,
        K: 'static + Hash + Eq + Debug + Clone,
        T: 'static + Debug + Clone + PartialEq {
        let key = try_as_ref!(key as *const K).clone();
        trans::SelectColumn::<M, T>::make(key).into()
    }
    let type_args = try_!(TypeArgs::parse(type_args, 3));
    dispatch!(monomorphize, [(type_args.0[0], @dist_dataset), (type_args.0[1], @hashable), (type_args.0[2], @primitives)], (key))
}

#[no_mangle]
pub extern "C" fn opendp_trans__make_clamp_vec(type_args: *const c_char, lower: *const c_void, upper: *const c_void) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<M, T>(lower: *const c_void, upper: *const c_void) -> FfiResult<*mut FfiTransformation>
        where M: 'static + Metric<Distance=u32> + Clone,
              T: 'static + Copy + PartialOrd {
        let lower = try_as_ref!(lower as *const T).clone();
        let upper = try_as_ref!(upper as *const T).clone();
        manipulation::Clamp::<M, Vec<T>>::make(lower, upper).into()
    }
    let type_args = try_!(TypeArgs::parse(type_args, 2));
    dispatch!(monomorphize, [(type_args.0[0], @dist_dataset), (type_args.0[1], @numbers)], (lower, upper))
}


#[no_mangle]
pub extern "C" fn opendp_trans__make_clamp_scalar(type_args: *const c_char, lower: *const c_void, upper: *const c_void) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<T, Q>(type_args: TypeArgs, lower: *const c_void, upper: *const c_void) -> FfiResult<*mut FfiTransformation>
        where T: 'static + Clone + PartialOrd,
              Q: 'static + One + Mul<Output=Q> + Div<Output=Q> + PartialOrd + DistanceCast {
        let lower = try_as_ref!(lower as *const T).clone();
        let upper = try_as_ref!(upper as *const T).clone();

        fn monomorphize2<M, T, Q>(lower: T, upper: T) -> FfiResult<*mut FfiTransformation>
            where M: 'static + SensitivityMetric<Distance=Q>,
                  T: 'static + Clone + PartialOrd,
                  Q: 'static + One + Mul<Output=Q> + Div<Output=Q> + PartialOrd + DistanceCast {
            trans::manipulation::Clamp::<M, T>::make(lower, upper).into()
        }
        dispatch!(monomorphize2, [
            (type_args.0[0], [L1Sensitivity<Q>, L2Sensitivity<Q>]),
            (type_args.0[2], [T]), (type_args.0[3], [Q])
        ], (lower, upper))
    }
    let type_args = try_!(TypeArgs::parse(type_args, 3));
    dispatch!(monomorphize, [(type_args.0[2], @numbers), (type_args.0[3], @numbers)], (type_args, lower, upper))
}


#[no_mangle]
pub extern "C" fn opendp_trans__make_cast_vec(type_args: *const c_char) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<M, TI, TO>() -> FfiResult<*mut FfiTransformation> where
        M: 'static + DatasetMetric<Distance=u32>, TI: 'static + Clone, TO: 'static + CastFrom<TI> + Default {
        trans::manipulation::Cast::<M, Vec<TI>, Vec<TO>>::make().into()
    }
    let type_args = try_!(TypeArgs::parse(type_args, 3));
    dispatch!(monomorphize, [(type_args.0[0], @dist_dataset), (type_args.0[1], @primitives), (type_args.0[2], @primitives)], ())
}

// #[no_mangle]
// pub extern "C" fn opendp_trans__make_cast(type_args: *const c_char) -> FfiResult<*mut FfiTransformation> {
//     fn monomorphize<TI, TO>(type_args: TypeArgs) -> FfiResult<*mut FfiTransformation>
//         where TI: 'static + Clone + DistanceCast,
//               TO: 'static + CastFrom<TI> + Default + DistanceCast + One + Div<Output=TO> + Mul<Output=TO> + PartialOrd {
//
//         fn monomorphize2<MI, MO, TI, TO>() -> FfiResult<*mut FfiTransformation>
//             where MI: 'static + SensitivityMetric<Distance=TI>,
//                   MO: 'static + SensitivityMetric<Distance=TO>,
//                   TI: 'static + Clone + DistanceCast,
//                   TO: 'static + CastFrom<TI> + Default + DistanceCast + One + Div<Output=TO> + Mul<Output=TO> + PartialOrd {
//             let transformation = trans::manipulation::Cast::<MI, MO, TI, TO>::make();
//             FfiResult::new(transformation.map(FfiTransformation::new_from_types))
//         }
//         dispatch!(monomorphize2, [
//             (type_args.0[0], [L1Sensitivity<TI>, L2Sensitivity<TI>]),
//             (type_args.0[1], [L1Sensitivity<TO>, L2Sensitivity<TO>]),
//             (type_args.0[2], [TI]), (type_args.0[3], [TO])
//         ], ())
//     }
//     let type_args = try_!(TypeArgs::parse(type_args, 4));
//     dispatch!(monomorphize, [(type_args.0[2], @numbers), (type_args.0[3], @numbers)], (type_args))
// }

#[no_mangle]
pub extern "C" fn opendp_trans__make_bounded_sum(type_args: *const c_char, lower: *const c_void, upper: *const c_void) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<T>(type_args: TypeArgs, lower: *const c_void, upper: *const c_void) -> FfiResult<*mut FfiTransformation>
        where T: 'static + Clone + PartialOrd + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Sum<T> + Abs + DistanceCast {
        fn monomorphize2<MI, MO, T>(lower: T, upper: T) -> FfiResult<*mut FfiTransformation>
            where MI: 'static + DatasetMetric<Distance=u32>,
                  MO: 'static + SensitivityMetric<Distance=T>,
                  T: 'static + Clone + PartialOrd + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Sum<T> + Abs + DistanceCast,
                  BoundedSum<MI, MO>: BoundedSumConstant<MI, MO> {
            sum::BoundedSum::<MI, MO>::make(lower, upper).into()
        }
        let lower = try_as_ref!(lower as *const T).clone();
        let upper = try_as_ref!(upper as *const T).clone();
        dispatch!(monomorphize2, [
            (type_args.0[0], [HammingDistance, SymmetricDistance]),
            (type_args.0[1], [L1Sensitivity<T>, L2Sensitivity<T>]),
            (type_args.0[2], [T])
        ], (lower, upper))
    }
    let type_args = try_!(TypeArgs::parse(type_args, 3));
    dispatch!(monomorphize, [(type_args.0[2], @numbers)], (type_args, lower, upper))
}


#[no_mangle]
pub extern "C" fn opendp_trans__make_bounded_sum_n(type_args: *const c_char, lower: *const c_void, upper: *const c_void, n: c_uint) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<T>(type_args: TypeArgs, lower: *const c_void, upper: *const c_void, n: usize) -> FfiResult<*mut FfiTransformation>
        where T: 'static + Copy + PartialOrd + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Sum<T> + DistanceCast + Abs {

        fn monomorphize2<MO, T>(lower: T, upper: T, n: usize) -> FfiResult<*mut FfiTransformation>
            where MO: 'static + SensitivityMetric<Distance=T>,
                  T: 'static + Clone + PartialOrd + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Sum<T> + DistanceCast {
            sum::BoundedSum::<SymmetricDistance, MO>::make3(lower, upper, n).into()
        }
        let lower = try_as_ref!(lower as *const T).clone();
        let upper = try_as_ref!(upper as *const T).clone();
        dispatch!(monomorphize2, [
            (type_args.0[0], [L1Sensitivity<T>, L2Sensitivity<T>]),
            (type_args.0[1], [T])
        ], (lower, upper, n))
    }
    let n = n as usize;
    let type_args = try_!(TypeArgs::parse(type_args, 2));
    dispatch!(monomorphize, [(type_args.0[1], @numbers)], (type_args, lower, upper, n))
}


#[no_mangle]
pub extern "C" fn opendp_trans__make_count(type_args: *const c_char) -> FfiResult<*mut FfiTransformation> {

    fn monomorphize<MI, MO, T: 'static>() -> FfiResult<*mut FfiTransformation>
        where MI: 'static + DatasetMetric<Distance=u32> + Clone,
              MO: 'static + SensitivityMetric<Distance=u32> + Clone {
        count::Count::<MI, MO, T>::make().into()
    }
    let type_args = try_!(TypeArgs::parse(type_args, 3));
    dispatch!(monomorphize, [
        (type_args.0[0], [SymmetricDistance, HammingDistance]),
        (type_args.0[1], [L1Sensitivity<u32>, L2Sensitivity<u32>]),
        (type_args.0[2], @primitives)
    ], ())
}


#[no_mangle]
pub extern "C" fn opendp_trans__make_count_by_categories(type_args: *const c_char, categories: *const FfiObject) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<QO>(type_args: TypeArgs, categories: *const FfiObject) -> FfiResult<*mut FfiTransformation>
        where QO: 'static + Clone + DistanceCast + Mul<Output=QO> + Div<Output=QO> + PartialOrd + FloatConst + One + NumCast {

        fn monomorphize2<MI, MO, TI, TO, QO>(categories: *const FfiObject) -> FfiResult<*mut FfiTransformation>
            where MI: 'static + DatasetMetric<Distance=u32>,
                  MO: 'static + SensitivityMetric<Distance=QO>,
                  TI: 'static + Eq + Hash + Clone,
                  TO: 'static + Integer + Zero + One + AddAssign,
                  QO: 'static + Clone + DistanceCast + Mul<Output=QO> + Div<Output=QO> + PartialOrd + FloatConst + One + NumCast,
                  CountByCategories<MI, MO, TI, TO>: CountByCategoriesConstant<MI, MO> {
            let categories = try_as_ref!(categories as *const Vec<TI>).clone();
            count::CountByCategories::<MI, MO, TI, TO>::make(categories).into()
        }
        dispatch!(monomorphize2, [
            (type_args.0[0], [HammingDistance, SymmetricDistance]),
            (type_args.0[1], [L1Sensitivity<QO>, L2Sensitivity<QO>]),
            (type_args.0[2], @hashable),
            (type_args.0[3], @integers),
            (type_args.0[4], [QO])
        ], (categories))
    }
    let type_args = try_!(TypeArgs::parse(type_args, 5));
    dispatch!(monomorphize, [(type_args.0[4], @floats)], (type_args, categories))
}

#[no_mangle]
pub extern "C" fn opendp_trans__make_count_by(type_args: *const c_char, n: c_uint) -> FfiResult<*mut FfiTransformation> {
    fn monomorphize<QO>(type_args: TypeArgs, n: usize) -> FfiResult<*mut FfiTransformation>
        where QO: 'static + Clone + DistanceCast + Mul<Output=QO> + Div<Output=QO> + PartialOrd + FloatConst + One + NumCast {

        fn monomorphize2<MI, MO, TI, TO, QO>(n: usize) -> FfiResult<*mut FfiTransformation>
            where MI: 'static + DatasetMetric<Distance=u32>,
                  MO: 'static + SensitivityMetric<Distance=QO>,
                  TI: 'static + Eq + Hash + Clone,
                  TO: 'static + Integer + Zero + One + AddAssign,
                  QO: 'static + Clone + DistanceCast + Mul<Output=QO> + Div<Output=QO> + PartialOrd + FloatConst + One + NumCast,
                  CountBy<MI, MO, TI, TO>: CountByConstant<MI, MO> {
            count::CountBy::<MI, MO, TI, TO>::make(n).into()
        }
        dispatch!(monomorphize2, [
            (type_args.0[0], [HammingDistance, SymmetricDistance]),
            (type_args.0[1], [L1Sensitivity<QO>, L2Sensitivity<QO>]),
            (type_args.0[2], @hashable),
            (type_args.0[3], @integers),
            (type_args.0[4], [QO])
        ], (n))
    }
    // TODO: drop type_args.0[4] by parsing inner type from type_args.0[1], once data loading PR is merged
    let n = n as usize;
    let type_args = try_!(TypeArgs::parse(type_args, 5));
    dispatch!(monomorphize, [(type_args.0[4], @floats)], (type_args, n))
}

#[no_mangle]
pub extern "C" fn opendp_trans__bootstrap() -> *const c_char {
    let spec =
r#"{
"functions": [
    { "name": "make_identity", "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_split_lines", "args": [ ["const char *", "selector"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_parse_series", "args": [ ["const char *", "selector"], ["bool", "impute"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_split_records", "args": [ ["const char *", "selector"], ["const char *", "separator"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_create_dataframe", "args": [ ["const char *", "selector"], ["FfiObject *", "col_names"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_split_dataframe", "args": [ ["const char *", "selector"], ["const char *", "separator"], ["FfiObject *", "col_names"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_parse_column", "args": [ ["const char *", "selector"], ["void *", "key"], ["bool", "impute"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_select_column", "args": [ ["const char *", "selector"], ["void *", "key"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_clamp_vec", "args": [ ["const char *", "selector"], ["void *", "lower"], ["void *", "upper"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_clamp_scalar", "args": [ ["const char *", "selector"], ["void *", "lower"], ["void *", "upper"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_cast_vec", "args": [ ["const char *", "selector"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_bounded_sum", "args": [ ["const char *", "selector"], ["void *", "lower"], ["void *", "upper"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_bounded_sum_n", "args": [ ["const char *", "selector"], ["void *", "lower"], ["void *", "upper"], ["unsigned int", "n"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_count", "args": [ ["const char *", "selector"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_count_by", "args": [ ["const char *", "selector"], ["unsigned int", "n"] ], "ret": "FfiResult<FfiTransformation *>" },
    { "name": "make_count_by_categories", "args": [ ["const char *", "selector"], ["FfiObject *", "categories"] ], "ret": "FfiResult<FfiTransformation *>" }
]
}"#;
    util::bootstrap(spec)
}

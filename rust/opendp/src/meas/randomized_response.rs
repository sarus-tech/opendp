use std::collections::HashSet;
use std::hash::Hash;

use indexmap::set::IndexSet;
use std::ops::Index;
use rand::Rng;

use crate::core::{Function, Measurement, PrivacyRelation};
use crate::dist::{MaxDivergence, SymmetricDistance};
use crate::dom::{AllDomain, SizedDomain, VectorDomain};
use crate::error::Fallible;
use crate::samplers::SampleBernoulli;

fn randomized_response_bool(arg: &bool, prob: f64, constant_time: bool) -> Fallible<bool> {
    let lie = bool::sample_bernoulli(prob, constant_time)?;
    Ok(if bool::sample_bernoulli(prob, constant_time)? { *arg } else { lie })
}

type RRDomain<T> = SizedDomain<VectorDomain<AllDomain<T>>>;

// useful paper: http://csce.uark.edu/~xintaowu/publ/DPL-2014-003.pdf
// p is probability that output is correct
// most tutorials describe two balanced coin flips, so p = .75, giving eps = ln(.75 / .25) = ln(3)
pub fn make_central_randomized_response_bool(
    prob: f64, constant_time: bool, row_size: usize
) -> Fallible<Measurement<RRDomain<bool>, RRDomain<bool>, SymmetricDistance, MaxDivergence<f64>>> {

    if !(0.5..1.0).contains(&prob) { return fallible!(FailedFunction, "probability must be within [0.5, 1)") }
    let domain = SizedDomain::new(VectorDomain::new_all(), row_size);

    Ok(Measurement::new(
        domain.clone(), domain,
        Function::new_fallible(move |arg: &Vec<bool>| arg.iter()
            .map(|v| randomized_response_bool(v, prob, constant_time))
            .collect()),
        SymmetricDistance::default(),
        MaxDivergence::default(),
        PrivacyRelation::new(move |d_in: &u32, d_out: &f64| *d_out >= f64::from(*d_in) * (prob / (1. - prob)).ln()),
    ))
}

pub fn make_local_randomized_response_bool(
    prob: f64, constant_time: bool
) -> Fallible<Measurement<AllDomain<bool>, AllDomain<bool>, SymmetricDistance, MaxDivergence<f64>>> {
    Ok(Measurement::new(
        AllDomain::new(),
        AllDomain::new(),
        Function::new_fallible(move |arg: &bool| randomized_response_bool(arg, prob, constant_time)),
        SymmetricDistance::default(),
        MaxDivergence::default(),
        // TODO: should we introduce a new input metric that has a () associated type?
        //       The input metric doesn't really fit :(
        PrivacyRelation::new(move |d_in: &u32, d_out: &f64| d_in == &0 && *d_out >= (prob / (1. - prob)).ln()),
    ))
}

pub fn make_central_randomized_response<T>(
    categories: HashSet<T>, prob: f64, constant_time: bool, row_size: usize
) -> Fallible<Measurement<RRDomain<T>, RRDomain<T>, SymmetricDistance, MaxDivergence<f64>>>
    where T: 'static + Clone + Eq + Hash {

    if !(0.5..1.0).contains(&prob) {return fallible!(FailedFunction, "probability must be within [0.5, 1)")}

    Ok(Measurement::new(
        SizedDomain::new(VectorDomain::new_all(), row_size),
        SizedDomain::new(VectorDomain::new_all(), row_size),
        Function::new_fallible(move |arg: &Vec<T>| {
            // clone categories from a hashset into a mutable indexmap
            let mut categories: IndexSet<T> = categories.iter().cloned().collect();
            let len = categories.len();
            // TODO: implement openssl CoreRng generator, implement a constant-time int sampler
            let mut rng = rand::thread_rng();

            arg.iter().map(|v| {
                let bernoulli = bool::sample_bernoulli(prob, constant_time)?;

                // if v in categories
                Ok(if let Some(index) = categories.get_index_of(v) {
                    // swap v to the end, and sample from the first len - 1 categories
                    categories.swap_indices(index, len - 1);
                    let lie = categories.index(rng.gen_range(0, len - 1));
                    if bernoulli { v } else { lie }
                } else {
                    categories.index(rng.gen_range(0, len - 1))
                }.clone())
            }).collect()
        }),
        SymmetricDistance::default(),
        MaxDivergence::default(),
        PrivacyRelation::new(move |d_in: &u32, d_out: &f64| *d_out >= f64::from(*d_in) * (prob / (1. - prob)).ln()),
    ))
}

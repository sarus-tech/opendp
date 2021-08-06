//! Various implementations of Metric/Measure (and associated Distance).

use std::marker::PhantomData;
use rug::float::Round;
use rug::ops::DivAssignRound;
use num::{One, Zero};
use std::ops::Add;
use std::cmp::Ordering;

use crate::core::{DatasetMetric, Measure, Metric, SensitivityMetric, PrivacyRelation};
use crate::traits::{Midpoint, Tolerance};
use crate::error::Fallible;
use crate::samplers::CastInternalReal;


/// Measures
#[derive(Clone)]
pub struct MaxDivergence<Q>(PhantomData<Q>);
impl<Q> Default for MaxDivergence<Q> {
    fn default() -> Self { MaxDivergence(PhantomData) }
}

impl<Q> PartialEq for MaxDivergence<Q> {
    fn eq(&self, _other: &Self) -> bool { true }
}

impl<Q: Clone> Measure for MaxDivergence<Q> {
    type Distance = Q;
}

#[derive(Clone)]
pub struct SmoothedMaxDivergence<Q>(PhantomData<Q>);

impl<Q> Default for SmoothedMaxDivergence<Q> {
    fn default() -> Self { SmoothedMaxDivergence(PhantomData) }
}

impl<Q> PartialEq for SmoothedMaxDivergence<Q> {
    fn eq(&self, _other: &Self) -> bool { true }
}

impl<Q: Clone> Measure for SmoothedMaxDivergence<Q> {
    type Distance = EpsilonDelta<Q>;
}

#[derive(Clone)]
pub struct FSmoothedMaxDivergence<Q>(PhantomData<Q>);
impl<Q> Default for FSmoothedMaxDivergence<Q> {
    fn default() -> Self { FSmoothedMaxDivergence(PhantomData) }
}

impl<Q> PartialEq for FSmoothedMaxDivergence<Q> {
    fn eq(&self, _other: &Self) -> bool { true }
}

impl<Q: Clone> Measure for FSmoothedMaxDivergence<Q> {
    type Distance = Vec<EpsilonDelta<Q>>;
}
const MAX_ITERATIONS: usize = 100;
use core::fmt::Debug;

impl<MI, Q> PrivacyRelation<MI, FSmoothedMaxDivergence<Q>>
     where MI: Metric,
           Q: Clone + One + Zero + Tolerance + Midpoint + PartialOrd + CastInternalReal,
           MI::Distance: Clone + CastInternalReal + One + Zero + Tolerance + Midpoint + PartialOrd + Copy {

    pub fn find_epsilon (&self, d_in: &MI::Distance, delta: Q) -> Fallible<Q> {
        let mut eps_min:rug::Float = Q::zero().into_internal();
        let mut eps:rug::Float = Q::one().into_internal();
        let delta:rug::Float = delta.into_internal();

        let two:rug::Float = Q::one().into_internal() + Q::one().into_internal();
        let tolerance: rug::Float = Q::TOLERANCE.into_internal();
        for _ in 0..MAX_ITERATIONS {

            let dout = vec![EpsilonDelta {
                epsilon: Q::from_internal(eps.clone()),
                delta: Q::from_internal(delta.clone()),
            }];
            if !self.eval(&d_in, &dout)? {
                eps = eps.clone() * two.clone();
            }

            else {
                let mut eps_mid = eps_min.clone() + eps.clone();
                eps_mid.div_assign_round(two.clone(), Round::Up);
                let dout = vec![EpsilonDelta {
                    epsilon: Q::from_internal(eps_mid.clone()),
                    delta: Q::from_internal(delta.clone()),
                }];
                if self.eval(&d_in, &dout)? {
                    eps = eps_mid.clone();
                } else {
                    eps_min = eps_mid.clone();
                }
                if eps <= tolerance.clone() + eps_min.clone() {
                    return Ok(Q::from_internal(eps))
                }
            }
        }
        let dout = vec![EpsilonDelta {
            epsilon: Q::from_internal(eps.clone()),
            delta: Q::from_internal(delta.clone()),
        }];
        if !self.eval(&d_in, &dout)? {
            return Ok(Q::from_internal(rug::Float::with_val( eps.prec(), 1./ 0.)))
        }
        Ok(Q::from_internal(eps))
    }

    pub fn find_delta (&self, d_in: &MI::Distance, epsilon: Q) -> Fallible<Q> {
        let mut delta_min:rug::Float = Q::zero().into_internal();
        let mut delta:rug::Float = Q::one().into_internal();
        let epsilon:rug::Float = epsilon.into_internal();

        let two:rug::Float = Q::one().into_internal() + Q::one().into_internal();
        let tolerance: rug::Float = Q::TOLERANCE.into_internal();
        for _ in 0..MAX_ITERATIONS {
            let dout = vec![EpsilonDelta {
                epsilon: Q::from_internal(epsilon.clone()),
                delta: Q::from_internal(delta.clone()),
            }];
            if !self.eval(&d_in, &dout)? {
                delta = delta.clone() * two.clone();
            }

            else {
                let mut delta_mid = delta_min.clone() + delta.clone();
                delta_mid.div_assign_round(two.clone(), Round::Up);
                let dout = vec![EpsilonDelta {
                    epsilon: Q::from_internal(epsilon.clone()),
                    delta: Q::from_internal(delta_mid.clone()),
                }];
                if self.eval(&d_in, &dout)? {
                    delta = delta_mid.clone();
                } else {
                    delta_min = delta_mid.clone();
                }
                if delta <= tolerance.clone() + delta_min.clone() {
                    return Ok(Q::from_internal(delta))
                }
            }
        }
        let dout = vec![EpsilonDelta {
            epsilon: Q::from_internal(epsilon.clone()),
            delta: Q::from_internal(delta.clone()),
        }];
        if !self.eval(&d_in, &dout)? {
            return Ok(Q::one())
        }
        Ok(Q::from_internal(delta))
    }

    pub fn find_epsilons_deltas (
        &self,
        npoints: u8,
        delta_min: Q,
        d_in: MI::Distance,
    ) -> Vec<EpsilonDelta<Q>> {
        let delta_max = Q::from_internal(rug::Float::with_val(4, 1.0));
        let max_epsilon_exp = self.find_epsilon(&d_in, delta_min).unwrap().into_internal().exp();
        let min_epsilon_exp = self.find_epsilon(&d_in, delta_max).unwrap().into_internal().exp();

        let step = (max_epsilon_exp.clone() - min_epsilon_exp.clone()) / rug::Float::with_val(4, npoints - 1);
        (0..npoints)
            .map(|i| Q::from_internal(
                (min_epsilon_exp.clone() + step.clone() * rug::Float::with_val(4, i)).ln()
            ))
            .map(|eps| EpsilonDelta{
                epsilon: eps.clone(),
                delta: self.find_delta(&d_in, eps.clone()).unwrap()
            })
            .rev()
            .collect()
    }

    pub fn find_epsilon_delta_family (
        &self,
        npoints: u8,
        delta_min: Q,
        d_in: MI::Distance
    ) -> Vec<EpsilonDelta<Q>> {
        let delta_max = Q::from_internal(rug::Float::with_val(4, 1.0));
        let min_epsilon_exp = self.find_epsilon(&d_in, delta_min).unwrap().into_internal().exp();
        let max_epsilon_exp = self.find_epsilon(&d_in, delta_max).unwrap().into_internal().exp();

        if min_epsilon_exp == rug::Float::with_val(min_epsilon_exp.prec(), 1. / 0.) {
            return vec![EpsilonDelta{
                epsilon: Q::from_internal(rug::Float::with_val(min_epsilon_exp.prec(), 1. / 0.)),
                delta: Q::one(),
            }]
        }

        let step = (max_epsilon_exp.clone() - min_epsilon_exp.clone()) / rug::Float::with_val(4, npoints - 1);
        (0..npoints)
            .map(|i| Q::from_internal(
                (min_epsilon_exp.clone() + step.clone() * rug::Float::with_val(4, i)).ln()
            ))
            .map(|eps| EpsilonDelta{
                epsilon: eps.clone(),
                delta: self.find_delta(&d_in, eps.clone()).unwrap()
            })
            .rev()
            .collect()
    }
}

/// Metrics
#[derive(Clone)]
pub struct SymmetricDistance;

impl Default for SymmetricDistance {
    fn default() -> Self { SymmetricDistance }
}

impl PartialEq for SymmetricDistance {
    fn eq(&self, _other: &Self) -> bool { true }
}

impl Metric for SymmetricDistance {
    type Distance = u32;
}

impl DatasetMetric for SymmetricDistance {}

#[derive(Clone)]
pub struct SubstituteDistance;

impl Default for SubstituteDistance {
    fn default() -> Self { SubstituteDistance }
}

impl PartialEq for SubstituteDistance {
    fn eq(&self, _other: &Self) -> bool { true }
}

impl Metric for SubstituteDistance {
    type Distance = u32;
}

impl DatasetMetric for SubstituteDistance {}

// Sensitivity in P-space
pub struct LpDistance<Q, const P: usize>(PhantomData<Q>);
impl<Q, const P: usize> Default for LpDistance<Q, P> {
    fn default() -> Self { LpDistance(PhantomData) }
}

impl<Q, const P: usize> Clone for LpDistance<Q, P> {
    fn clone(&self) -> Self { Self::default() }
}
impl<Q, const P: usize> PartialEq for LpDistance<Q, P> {
    fn eq(&self, _other: &Self) -> bool { true }
}
impl<Q, const P: usize> Metric for LpDistance<Q, P> {
    type Distance = Q;
}
impl<Q, const P: usize> SensitivityMetric for LpDistance<Q, P> {}

pub type L1Distance<Q> = LpDistance<Q, 1>;
pub type L2Distance<Q> = LpDistance<Q, 2>;


pub struct AbsoluteDistance<Q>(PhantomData<Q>);
impl<Q> Default for AbsoluteDistance<Q> {
    fn default() -> Self { AbsoluteDistance(PhantomData) }
}

impl<Q> Clone for AbsoluteDistance<Q> {
    fn clone(&self) -> Self { Self::default() }
}
impl<Q> PartialEq for AbsoluteDistance<Q> {
    fn eq(&self, _other: &Self) -> bool { true }
}
impl<Q> Metric for AbsoluteDistance<Q> {
    type Distance = Q;
}
impl<Q> SensitivityMetric for AbsoluteDistance<Q> {}




#[derive(Debug)]
pub struct EpsilonDelta<T: Sized>{pub epsilon: T, pub delta: T}

// Derive annotations force traits to be present on the generic
impl<T: PartialOrd> PartialOrd for EpsilonDelta<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let epsilon_ord = self.epsilon.partial_cmp(&other.epsilon);
        let delta_ord = self.delta.partial_cmp(&other.delta);
        if epsilon_ord == delta_ord { epsilon_ord } else { None }
    }
}
impl<T: Clone> Clone for EpsilonDelta<T> {
    fn clone(&self) -> Self {
        EpsilonDelta {epsilon: self.epsilon.clone(), delta: self.delta.clone()}
    }
}
impl<T: PartialEq> PartialEq for EpsilonDelta<T> {
    fn eq(&self, other: &Self) -> bool {
        self.epsilon == other.epsilon && self.delta == other.delta
    }
}
impl<T: Zero + Sized + Add<Output=T> + Clone> Zero for EpsilonDelta<T> {
    fn zero() -> Self {
        EpsilonDelta { epsilon: T::zero(), delta: T::zero() }
    }
    fn is_zero(&self) -> bool {
        self.epsilon.is_zero() && self.delta.is_zero()
    }
}
impl<T: Add<Output=T> + Clone> Add<EpsilonDelta<T>> for EpsilonDelta<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        EpsilonDelta {epsilon: self.epsilon + rhs.epsilon, delta: self.delta + rhs.delta}
    }
}

#[derive(Debug)]
pub struct AlphaBeta{pub alpha: rug::Float, pub beta: rug::Float}

impl Clone for AlphaBeta {
    fn clone(&self) -> Self {
        AlphaBeta {alpha: self.alpha.clone(), beta: self.beta.clone()}
    }
}
impl PartialEq for AlphaBeta {
    fn eq(&self, other: &Self) -> bool {
        self.alpha == other.alpha //&& self.beta == other.beta
    }
}
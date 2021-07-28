use num::{Float, One};

use crate::core::{Function, Measure, Measurement, Metric, PrivacyRelation, Domain, SensitivityMetric};
use crate::dist::{L1Distance, MaxDivergence, FSmoothedMaxDivergence, AbsoluteDistance, EpsilonDelta};
use crate::dom::{AllDomain, VectorDomain};
use crate::samplers::{SampleLaplace, CastInternalReal};
use crate::error::*;
use crate::traits::{DistanceConstant, DistanceCast};
use crate::chain::BasicCompositionDistance;

pub trait LaplaceDomain: Domain {
    type Metric: SensitivityMetric<Distance=Self::Atom> + Default;
    type Atom;
    fn new() -> Self;
    fn noise_function(scale: Self::Atom) -> Function<Self, Self>;
}

impl<T> LaplaceDomain for AllDomain<T>
    where T: 'static + SampleLaplace + Float + DistanceCast {
    type Metric = AbsoluteDistance<T>;
    type Atom = Self::Carrier;

    fn new() -> Self { AllDomain::new() }
    fn noise_function(scale: Self::Carrier) -> Function<Self, Self> {
        Function::new_fallible(move |arg: &Self::Carrier| Self::Carrier::sample_laplace(*arg, scale, false))
    }
}

impl<T> LaplaceDomain for VectorDomain<AllDomain<T>>
    where T: 'static + SampleLaplace + Float + DistanceCast {
    type Metric = L1Distance<T>;
    type Atom = T;

    fn new() -> Self { VectorDomain::new_all() }
    fn noise_function(scale: T) -> Function<Self, Self> {
        Function::new_fallible(move |arg: &Self::Carrier| arg.iter()
            .map(|v| T::sample_laplace(*v, scale, false))
            .collect())
    }
}

pub trait LaplacePrivacyRelation<MI: Metric>: Measure {
    fn privacy_relation(scale: MI::Distance) -> PrivacyRelation<MI, Self>;
}

impl<MI: Metric> LaplacePrivacyRelation<MI> for MaxDivergence<MI::Distance>
    where MI::Distance: 'static + Clone + SampleLaplace + Float + DistanceConstant,
          MI: SensitivityMetric {
    fn privacy_relation(scale: MI::Distance) -> PrivacyRelation<MI, Self>{
        PrivacyRelation::new_from_constant(scale.recip())
    }
}

pub fn compute_dual_epsilon_delta<Q: 'static + Float + Clone + CastInternalReal > (scale: Q, epsilon: Q) -> EpsilonDelta<Q> { // implement trait
    use rug::float::Round;
    let mut scale_float: rug::Float = scale.clone().into_internal();
    scale_float.recip_round(Round::Up); // scale_float -> 1 / scale_float
    let epsilon_float: rug::Float = epsilon.clone().into_internal();
    let mut res = (epsilon_float - scale_float) / (2.).into_internal();
    res.exp_round(Round::Down); // res =  exp((epsilon - 1 / scale) / 2)
    let delta = Q::one() - Q::from_internal(res); // delta = 1 - exp((epsilon - 1 / scale) / 2)
    EpsilonDelta {
        epsilon: epsilon,
        delta: delta,
    }
}

//#[cfg(feature="use-mpfr")]
impl<MI: Metric> LaplacePrivacyRelation<MI> for FSmoothedMaxDivergence<MI::Distance>
    where MI::Distance: 'static + Clone + Float + One + CastInternalReal,
          MI: SensitivityMetric {
    fn privacy_relation(scale: MI::Distance) -> PrivacyRelation<MI, Self> {
        PrivacyRelation::new_fallible(move |d_in: &MI::Distance, d_out: &Vec<EpsilonDelta<MI::Distance>>| {
            if d_in.is_sign_negative() {
                return fallible!(InvalidDistance, "laplace mechanism: input sensitivity must be non-negative")
            }

            let mut result = true;
            for EpsilonDelta { epsilon, delta } in d_out {
                if epsilon.is_sign_negative() {
                    return fallible!(InvalidDistance, "laplace mechanism: epsilon must be positive or 0")
                }
                if delta.is_sign_negative() {
                    return fallible!(InvalidDistance, "laplace mechanism: delta must be positive or 0")
                }

                let delta_dual = compute_dual_epsilon_delta(scale, *epsilon).delta;
                result = result & (delta >= &delta_dual);
                if result == false {
                    break;
                }
            }
            Ok(result)
        })
    }
}

pub fn make_base_laplace<D, MO>(scale: D::Atom) -> Fallible<Measurement<D, D, D::Metric, MO>>
    where D: LaplaceDomain,
          D::Atom: 'static + Clone + SampleLaplace + Float + DistanceCast + BasicCompositionDistance,
          MO: Measure + LaplacePrivacyRelation<D::Metric> {
    if scale.is_sign_negative() {
        return fallible!(MakeMeasurement, "scale must not be negative")
    }
    Ok(Measurement::new(
        D::new(),
        D::new(),
        D::noise_function(scale.clone()),
        D::Metric::default(),
        MO::default(),
        MO::privacy_relation(scale),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trans::make_bounded_mean;

    #[test]
    fn test_chain_laplace() -> Fallible<()> {
        let chain = (
            make_bounded_mean(10.0, 12.0, 3)? >>
            make_base_laplace::<AllDomain<_>, MaxDivergence<_>>(1.0)?
        )?;
        let _ret = chain.function.eval(&vec![10.0, 11.0, 12.0])?;
        Ok(())
    }

    #[test]
    fn test_make_laplace_mechanism() -> Fallible<()> {
        let measurement = make_base_laplace::<AllDomain<_>, MaxDivergence<_>>(1.0)?;
        let _ret = measurement.function.eval(&0.0)?;
        assert!(measurement.privacy_relation.eval(&1., &1.)?);

        let measurement = make_base_laplace::<VectorDomain<_>, FSmoothedMaxDivergence<_>>(1.0)?;
        let d_out = vec![
            EpsilonDelta { epsilon: 0., delta: 0.4 },
            EpsilonDelta { epsilon: 0.25, delta: 0.32 },
            EpsilonDelta { epsilon: 0.5, delta: 0.23 },
            EpsilonDelta { epsilon: 0.75, delta: 0.12 },
            EpsilonDelta { epsilon: 1., delta: 0.0 },
        ];
        assert!(measurement.privacy_relation.eval(&1.0, &d_out)?);
        Ok(())
    }

    #[test]
    fn test_make_vector_laplace_mechanism() -> Fallible<()> {
        let measurement = make_base_laplace::<VectorDomain<_>, MaxDivergence<_>>(1.0)?;
        let arg = vec![1.0, 2.0, 3.0];
        let _ret = measurement.function.eval(&arg)?;

        assert!(measurement.privacy_relation.eval(&1., &1.)?);
        Ok(())
    }
}


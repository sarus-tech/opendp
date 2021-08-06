use num::{Zero, One, Float};
use std::ops::{Shr, Sub, Div};
use std::fmt::Debug; // TODO: rm that

use crate::core::{Domain, Function, HintMt, HintTt, Measure, Measurement, Metric, PrivacyRelation, StabilityRelation, Transformation};
use crate::dom::{PairDomain, VectorDomain};
use crate::error::Fallible;
use crate::traits::{MetricDistance, MeasureDistance, Midpoint, FallibleSub, Tolerance};
use crate::dist::{MaxDivergence, SmoothedMaxDivergence, FSmoothedMaxDivergence, EpsilonDelta, AlphaBeta};
use crate::samplers::CastInternalReal;

pub fn make_chain_mt<DI, DX, DO, MI, MX, MO>(
    measurement1: &Measurement<DX, DO, MX, MO>,
    transformation0: &Transformation<DI, DX, MI, MX>,
    hint: Option<&HintMt<MI, MO, MX>>,
) -> Fallible<Measurement<DI, DO, MI, MO>>
    where DI: 'static + Domain,
          DX: 'static + Domain,
          DO: 'static + Domain,
          MI: 'static + Metric,
          MX: 'static + Metric,
          MO: 'static + Measure {
    if transformation0.output_domain != measurement1.input_domain {
        return fallible!(DomainMismatch, "Intermediate domain mismatch");
    } else if transformation0.output_metric != measurement1.input_metric {
        return fallible!(MetricMismatch, "Intermediate metric mismatch");
    }

    Ok(Measurement::new(
        transformation0.input_domain.clone(),
        measurement1.output_domain.clone(),
        Function::make_chain(&measurement1.function, &transformation0.function),
        transformation0.input_metric.clone(),
        measurement1.output_measure.clone(),
        PrivacyRelation::make_chain(&measurement1.privacy_relation,&transformation0.stability_relation, hint)
    ))
}

pub fn make_chain_tt<DI, DX, DO, MI, MX, MO>(
    transformation1: &Transformation<DX, DO, MX, MO>,
    transformation0: &Transformation<DI, DX, MI, MX>,
    hint: Option<&HintTt<MI, MO, MX>>,
) -> Fallible<Transformation<DI, DO, MI, MO>>
    where DI: 'static + Domain,
          DX: 'static + Domain,
          DO: 'static + Domain,
          MI: 'static + Metric,
          MX: 'static + Metric,
          MO: 'static + Metric {
    if transformation0.output_domain != transformation1.input_domain {
        return fallible!(DomainMismatch, "Intermediate domain mismatch");
    } else if transformation0.output_metric != transformation1.input_metric {
        return fallible!(MetricMismatch, "Intermediate metric mismatch");
    }

    Ok(Transformation::new(
        transformation0.input_domain.clone(),
        transformation1.output_domain.clone(),
        Function::make_chain(&transformation1.function, &transformation0.function),
        transformation0.input_metric.clone(),
        transformation1.output_metric.clone(),
        StabilityRelation::make_chain(&transformation1.stability_relation,&transformation0.stability_relation, hint)
    ))
}

pub trait BasicComposition<MI: Metric>: Measure {
    fn basic_composition(
        &self,
        relations: &Vec<PrivacyRelation<MI, Self>>,
        d_in: &MI::Distance, d_out: &Self::Distance
    ) -> Fallible<bool>;
}

pub trait BasicCompositionDistance: MeasureDistance + Clone + Midpoint + Zero + Tolerance {
    type Atom: Sub<Output=Self::Atom> + One + Div<Output=Self::Atom>;
}

impl BasicCompositionDistance for f64 {type Atom = f64;}
impl BasicCompositionDistance for f32 {type Atom = f64;}
impl<T> BasicCompositionDistance for EpsilonDelta<T>
    where T: for<'a> Sub<&'a T, Output=T> + Sub<Output=T> + One + Div<Output=T> + PartialOrd + Tolerance + Zero + Clone {
    type Atom = T;
}

// impl<Q: MeasureDistance + Clone + FallibleSub<Output=Q> + Midpoint + Zero + Tolerance + PartialOrd> BasicCompositionDistance for Q {}

impl<MI: Metric, Q: Clone> BasicComposition<MI> for MaxDivergence<Q>
    where Q: BasicCompositionDistance<Atom=Q>,
          MI::Distance: Clone {
    fn basic_composition(
        &self, relations: &Vec<PrivacyRelation<MI, Self>>,
        d_in: &MI::Distance, d_out: &Self::Distance
    ) -> Fallible<bool> {
        basic_composition(relations, d_in, d_out)
    }
}

impl<MI: Metric, Q: Clone> BasicComposition<MI> for SmoothedMaxDivergence<Q>
    where EpsilonDelta<Q>: BasicCompositionDistance<Atom=Q>,
          MI::Distance: Clone {
    fn basic_composition(
        &self, relations: &Vec<PrivacyRelation<MI, Self>>,
        d_in: &MI::Distance, d_out: &Self::Distance
    ) -> Fallible<bool> {
        basic_composition(relations, d_in, d_out)
    }
}

fn basic_composition<MI: Metric, MO: Measure>(
    relations: &Vec<PrivacyRelation<MI, MO>>,
    d_in: &MI::Distance,
    d_out: &MO::Distance
) -> Fallible<bool>
    where MO::Distance: BasicCompositionDistance,
          MI::Distance: Clone {
    let mut d_out = d_out.clone();

    for relation in relations {
        if let Some(usage) = basic_composition_binary_search(
            d_in.clone(), d_out.clone(), relation)? {

            d_out = d_out.sub(&usage)?;
        } else {
            return Ok(false)
        }
    }
    Ok(true)
}

pub fn make_basic_composition_multi<DI, DO, MI, MO>(
    measurements: &Vec<&Measurement<DI, DO, MI, MO>>
) -> Fallible<Measurement<DI, VectorDomain<DO>, MI, MO>>
    where DI: 'static + Domain,
          DO: 'static + Domain,
          MI: 'static + Metric,
          MI::Distance: 'static + MetricDistance + Clone,
          MO: 'static + Measure + BasicComposition<MI>,
          MO::Distance: 'static + MeasureDistance + Clone {

    if measurements.is_empty() {
        return fallible!(MakeMeasurement, "Must have at least one measurement")
    }

    let input_domain = measurements[0].input_domain.clone();
    let output_domain = measurements[0].output_domain.clone();
    let input_metric = measurements[0].input_metric.clone();
    let output_measure = measurements[0].output_measure.clone();

    if !measurements.iter().all(|v| input_domain == v.input_domain) {
        return fallible!(DomainMismatch, "All input domains must be the same");
    }
    if !measurements.iter().all(|v| output_domain == v.output_domain) {
        return fallible!(DomainMismatch, "All output domains must be the same");
    }
    if !measurements.iter().all(|v| input_metric == v.input_metric) {
        return fallible!(MetricMismatch, "All input metrics must be the same");
    }
    if !measurements.iter().all(|v| output_measure == v.output_measure) {
        return fallible!(MetricMismatch, "All output measures must be the same");
    }

    let mut functions = Vec::new();
    let mut relations = Vec::new();
    for measurement in measurements {
        functions.push(measurement.function.clone());
        relations.push(measurement.privacy_relation.clone());
    }

    Ok(Measurement::new(
        input_domain,
        VectorDomain::new(output_domain),
        Function::new_fallible(move |arg: &DI::Carrier|
            functions.iter().map(|f| f.eval(arg)).collect()),
        input_metric,
        output_measure.clone(),
        PrivacyRelation::new_fallible(move |d_in: &MI::Distance, d_out: &MO::Distance| {
            output_measure.basic_composition(&relations, d_in, d_out)
        })
    ))
}

const MAX_ITERATIONS: usize = 100;

fn basic_composition_binary_search<MI, MO>(
    d_in: MI::Distance, mut d_out: MO::Distance,
    predicate: &PrivacyRelation<MI, MO>
) -> Fallible<Option<MO::Distance>>
    where MI: Metric,
          MO: Measure,
          MO::Distance: Midpoint + Zero + Clone + Tolerance + PartialOrd {

    // d_out is d_max, we use binary search to reduce d_out
    // to the smallest value that still passes the relation
    if !predicate.eval(&d_in, &d_out)? {
        return Ok(None)
    }

    let mut d_min = MO::Distance::zero();
    for _ in 0..MAX_ITERATIONS {
        let d_mid = d_min.clone().midpoint(d_out.clone());

        if predicate.eval(&d_in, &d_mid)? {
            d_out = d_mid;
        } else {
            d_min = d_mid;
        }
        if d_out <= MO::Distance::TOLERANCE + d_min.clone() { return Ok(Some(d_out)) }
    }
    Ok(Some(d_out))
}

pub fn make_basic_composition<DI, DO0, DO1, MI, MO>(measurement0: &Measurement<DI, DO0, MI, MO>, measurement1: &Measurement<DI, DO1, MI, MO>) -> Fallible<Measurement<DI, PairDomain<DO0, DO1>, MI, MO>>
    where DI: 'static + Domain,
          DO0: 'static + Domain,
          DO1: 'static + Domain,
          MI: 'static + Metric,
          MO: 'static + Measure {
    if measurement0.input_domain != measurement1.input_domain {
        return fallible!(DomainMismatch, "Input domain mismatch");
    } else if measurement0.input_metric != measurement1.input_metric {
        return fallible!(MetricMismatch, "Input metric mismatch");
    } else if measurement0.output_measure != measurement1.output_measure {
        return fallible!(MeasureMismatch, "Output measure mismatch");
    }

    Ok(Measurement::new(
        measurement0.input_domain.clone(),
        PairDomain::new(measurement0.output_domain.clone(), measurement1.output_domain.clone()),
        Function::make_basic_composition(&measurement0.function, &measurement1.function),
        measurement0.input_metric.clone(),
        measurement0.output_measure.clone(),
        // TODO: PrivacyRelation for make_composition
        PrivacyRelation::new(|_i, _o| false),
    ))
}

// Bounded complexity composition
#[derive(Debug)]
pub struct ProbabilityLogRatio{pub probability: rug::Float, pub logratio: rug::Float}

impl Clone for ProbabilityLogRatio {
    fn clone(&self) -> Self {
        ProbabilityLogRatio {probability: self.probability.clone(), logratio: self.logratio.clone()}
    }
}

#[derive(Debug)]
pub struct ProbabilitiesLogRatios {
    probabilities: Vec<rug::Float>,
    logratios: Vec<rug::Float>,
}

impl Clone for ProbabilitiesLogRatios {
    fn clone(&self) -> Self {
        ProbabilitiesLogRatios {probabilities: self.probabilities.clone(), logratios: self.logratios.clone()}
    }
}

impl ProbabilitiesLogRatios {

    pub fn new_empty () -> Self {
        ProbabilitiesLogRatios {
            probabilities: Vec::<rug::Float>::new(),
            logratios: Vec::<rug::Float>::new(),
        }
    }

    pub fn new (probas: Vec<rug::Float>, ratios: Vec<rug::Float>) -> Self {
        ProbabilitiesLogRatios {
            probabilities: probas,
            logratios: ratios,
        }
    }

    pub fn log_sum_exp(logs: &Vec<rug::Float>) -> rug::Float{
        let precision = logs.clone()[0].prec();
        let mut m = logs[0].clone();
        for log in logs {
            if log > &m {
                m = log.clone();
            }
        }
        let neg_infinity = rug::Float::with_val(precision, 0.0).ln();
        if m == neg_infinity {
            m
        } else {
            let modified_logs = rug::Float::with_val(
                precision,
                rug::Float::sum(
                    logs.clone().iter()
                        .map(|x| (x - m.clone()).exp())
                        .collect::<Vec<rug::Float>>()
                        .iter()
                    )
            ).ln();
            modified_logs + m
        }
    }

    pub fn new_from_logproba(logdensity: Vec<rug::Float>) -> Self {
        let m = Self::log_sum_exp(&logdensity);
        let logdensity = logdensity.iter()
            .map(|x| x - m.clone());
        let density: Vec<rug::Float> = logdensity.clone().map(|x| x.exp()).collect();
        let ratios: Vec<rug::Float> = logdensity.clone()
            .zip(logdensity.clone().rev())
            .map(|(x, y)| x-y)
            .collect();
        Self::new(density, ratios)
    }

    pub fn push(&mut self, proba: rug::Float, logratio: rug::Float) -> () {
        self.probabilities.push(proba);
        self.logratios.push(logratio);
    }

    pub fn normalize(&mut self) -> () {
        let precision = &self.probabilities[0].prec(); // TODO

        let mut proba_log_ratio_vec = self.to_proba_logratio_vec();
        proba_log_ratio_vec.sort_by(|a, b| a.logratio.partial_cmp(&b.logratio).unwrap());
        proba_log_ratio_vec.reverse();

        let mut current_ratio = proba_log_ratio_vec[0].logratio.clone();
        let mut current_probas: Vec<rug::Float> = Vec::new();
        *self = Self::new_empty();
        for logratio_prob in &proba_log_ratio_vec {
            if logratio_prob.logratio != current_ratio {
                self.push(
                    rug::Float::with_val(*precision, rug::Float::sum(current_probas.clone().iter())),
                    current_ratio
                );
                current_probas = Vec::new();
            }
            current_probas.push(logratio_prob.probability.clone());
            current_ratio = logratio_prob.logratio.clone();
        }
        self.push(
            rug::Float::with_val(*precision, rug::Float::sum(current_probas.clone().iter())),
            current_ratio
        );
    }

    pub fn to_proba_logratio_vec (&self) -> Vec<ProbabilityLogRatio> {
        self.probabilities.clone().iter()
            .zip(self.logratios.clone().iter())
            .map(|(p, r)| ProbabilityLogRatio{probability:p.clone(), logratio:r.clone()})
            .collect()
    }

    pub fn to_alpha_beta_vec (&self) -> Vec<AlphaBeta> {
        let precision = &self.probabilities[0].prec();
        let one: rug::Float = rug::Float::with_val(*precision, 1.);
        let zero: rug::Float = rug::Float::with_val(*precision, 0.);

        let vec_proba_log_ratio = self.to_proba_logratio_vec();
        let mut alpha_beta_vec: Vec<AlphaBeta> = Vec::new();
        alpha_beta_vec.push(AlphaBeta{alpha: zero.clone(), beta: one.clone()});
        for threshold in self.clone().logratios {
            let alpha = rug::Float::with_val(
                *precision,
                rug::Float::sum(
                    vec_proba_log_ratio.iter()
                        .map(|ProbabilityLogRatio{probability:p, logratio:r}| {if r <= &threshold {p.clone()} else {zero.clone()}})
                        .collect::<Vec<rug::Float>>()
                        .iter()
                    )
            );

            let beta = rug::Float::with_val(
                *precision,
                rug::Float::sum(
                    vec_proba_log_ratio.iter()
                        .map(|ProbabilityLogRatio{probability:p, logratio:r}| {if r > &threshold {p.clone() * (-r.clone()).exp()} else {zero.clone()}})
                        .collect::<Vec<rug::Float>>()
                        .iter()
                    )
            );
            alpha_beta_vec.push(AlphaBeta{alpha: alpha, beta: beta});
        };
        alpha_beta_vec.sort_by(|a, b| a.alpha.partial_cmp(&b.alpha).unwrap());
        alpha_beta_vec
    }

    pub fn from_alpha_beta_vec (alphas_betas: &Vec<AlphaBeta>) -> Self { // compute_adjacent_probabilities
        let mut proba_vec: Vec<rug::Float> = Vec::new();
        let size = alphas_betas.iter().len();
        for i in 1..size { // TODO: optimize that with a map
            proba_vec.push(alphas_betas[i].alpha.clone() - alphas_betas[i-1].alpha.clone());
            }
        let ratio_vec: Vec<rug::Float> = proba_vec.iter()
            .zip(proba_vec.iter().rev())
            .map(|(p,q)| p.clone().ln() - q.clone().ln())
            .collect();
        Self::new(proba_vec, ratio_vec)
    }

    pub fn from_relation <MI: Metric> (
        relation: PrivacyRelation<MI, FSmoothedMaxDivergence<MI::Distance>>,
        npoints: u8,
        delta_min: MI::Distance,
    ) -> Self
    where MI::Distance: Clone + CastInternalReal + One + Zero + Tolerance + Midpoint + PartialOrd + Copy + Float + Debug
    {

        let epsilon_deltas = relation.find_epsilon_delta_family(
            npoints.clone(),
            delta_min.clone(),
            MI::Distance::one()
        );

        let alphas_betas_vec =  TradeoffFunc::from_epsilon_delta_vec(&epsilon_deltas).to_alpha_beta_vec();
        let mut probas_log_ratios = ProbabilitiesLogRatios::from_alpha_beta_vec(&alphas_betas_vec);
        probas_log_ratios.normalize();
        probas_log_ratios
    }

    pub fn len (&self) -> usize {
        self.probabilities.clone().iter().len()
    }

    pub fn sum_probas (&self) -> rug::Float {
        let precision = self.probabilities[0].clone().prec();
        rug::Float::with_val(
            precision,
            rug::Float::sum(self.probabilities.clone().iter())
        )
    }

    pub fn sum_probas_q (&self) -> rug::Float {
        let precision = self.probabilities[0].clone().prec();
        rug::Float::with_val(
            precision,
            rug::Float::sum(
                self.probabilities.clone().iter()
                    .zip(self.logratios.clone().iter())
                    .map(|(p, logr)| p.clone() / logr.clone().exp())
                    .collect::<Vec<rug::Float>>()
                    .iter()
            )
        )
    }

    pub fn log_proba(&self) -> Vec<rug::Float> {
        self.probabilities.iter()
            .map(|x| x.clone().ln())
            .collect()
    }

    pub fn compose(&self, probas_log_ratios: Self) -> Self {
        let mut compo_log_proba: Vec<rug::Float> = Vec::new();
        for log_prob1 in &self.log_proba() {
            for log_prob2 in &probas_log_ratios.log_proba() {
                compo_log_proba.push(log_prob1.clone() + log_prob2.clone());
            }
        }
        let mut compo_proba_log_ratio = ProbabilitiesLogRatios::new_from_logproba(compo_log_proba);
        compo_proba_log_ratio.normalize();
        compo_proba_log_ratio
    }
}

#[derive(Debug)]
pub struct TradeoffFunc {
    alphas: Vec<rug::Float>,
    betas: Vec<rug::Float>,
}

impl Clone for TradeoffFunc {
    fn clone(&self) -> Self {
        TradeoffFunc {alphas: self.alphas.clone(), betas: self.betas.clone()}
    }
}

impl TradeoffFunc {
    pub fn new_empty () -> Self {
        TradeoffFunc {
            alphas: Vec::<rug::Float>::new(),
            betas: Vec::<rug::Float>::new(),
        }
    }

    pub fn new (alphas: Vec<rug::Float>, betas: Vec<rug::Float>) -> Self {
        TradeoffFunc {
            alphas: alphas,
            betas: betas,
        }
    }

    pub fn push(&mut self, alpha: rug::Float, beta: rug::Float) -> () {
        self.alphas.push(alpha);
        self.betas.push(beta);
    }

    pub fn from_epsilon_delta_vec <Q> (epsilons_deltas: &Vec<EpsilonDelta<Q>>) -> Self
        where Q:  Zero + One + Clone + Sub + Float + CastInternalReal + Debug{
        let one: rug::Float = Q::one().into_internal();
        let zero: rug::Float = Q::zero().into_internal();

        let mut alpha_beta_vec = vec![
            AlphaBeta {alpha: zero.clone(), beta: one.clone() - epsilons_deltas[0].delta.into_internal()}
            ];
        let size = epsilons_deltas.iter().len();
        for i in 1..size {
            let alpha =
                (epsilons_deltas[i-1].delta.into_internal() - epsilons_deltas[i].delta.into_internal())
                /
                (epsilons_deltas[i].epsilon.into_internal().exp() - epsilons_deltas[i-1].epsilon.into_internal().exp());

            let beta = (
                    epsilons_deltas[i].epsilon.into_internal().exp() *(one.clone() - epsilons_deltas[i-1].delta.into_internal())
                    -
                    epsilons_deltas[i-1].epsilon.into_internal().exp() *(one.clone() - epsilons_deltas[i].delta.into_internal())
                )
                /
                (epsilons_deltas[i].epsilon.into_internal().exp() - epsilons_deltas[i-1].epsilon.into_internal().exp());
            alpha_beta_vec.push(AlphaBeta {alpha: alpha, beta: beta});
        }

        let mut rev_alpha_beta_vec = alpha_beta_vec.iter()
            .map(|alpha_beta| AlphaBeta {alpha: alpha_beta.beta.clone(), beta: alpha_beta.alpha.clone()})
            .rev()
            .collect();
        alpha_beta_vec.append(&mut rev_alpha_beta_vec);
        alpha_beta_vec.sort_by(|a, b| b.alpha.partial_cmp(&a.alpha).unwrap());
        alpha_beta_vec.reverse();
        Self::from_alpha_beta_vec(alpha_beta_vec)
    }

    pub fn to_alpha_beta_vec (&self) -> Vec<AlphaBeta> {
        self.alphas.clone().iter()
            .zip(self.betas.clone().iter())
            .map(|(a, b)| AlphaBeta{alpha:a.clone(), beta:b.clone()})
            .collect()
    }

    pub fn from_alpha_beta_vec (alphas_betas_vec: Vec<AlphaBeta>) -> Self {
        let alpha_vec:Vec<rug::Float> = alphas_betas_vec.iter().map(|x| x.alpha.clone()).collect();
        let beta_vec:Vec<rug::Float> = alphas_betas_vec.iter().map(|x| x.beta.clone()).collect();
        Self::new(alpha_vec, beta_vec)
    }

    fn to_epsilon_delta <Q: CastInternalReal + Clone> (
        &self,
        epsilon: Q,
    ) -> EpsilonDelta<Q> {
        let precision = self.clone().alphas[0].prec();
        let one: rug::Float = rug::Float::with_val(precision, 1.);
        let zero: rug::Float = rug::Float::with_val(precision, 0.);

        let mut delta = self.to_alpha_beta_vec()
            .iter()
            .map(|AlphaBeta{alpha: a, beta: b}| one.clone() - a.clone() * epsilon.clone().into_internal().exp() - b.clone())
            .max_by_key(|a| rug::float::OrdFloat::from(a.clone()))
            .unwrap();

        if delta < zero.clone() {
            delta = zero.clone();
        }

        EpsilonDelta {
            epsilon: epsilon,
            delta: Q::from_internal(delta),
        }
    }
}

pub fn bounded_complexity_composition_privacy_relation <MI: Metric>(
    relation1: PrivacyRelation<MI, FSmoothedMaxDivergence<MI::Distance>>,
    relation2: PrivacyRelation<MI, FSmoothedMaxDivergence<MI::Distance>>,
    npoints: u8,
    delta_min: MI::Distance,
) -> PrivacyRelation<MI, FSmoothedMaxDivergence<MI::Distance>>
    where MI::Distance: Clone + CastInternalReal + One + Zero + Tolerance + Midpoint + PartialOrd + Float + Debug {

    let proba_log_ratio1 = ProbabilitiesLogRatios::from_relation(relation1, npoints.clone(), delta_min.clone());
    let proba_log_ratio2 = ProbabilitiesLogRatios::from_relation(relation2, npoints.clone(), delta_min.clone());
    let compo_proba_log_ratio = proba_log_ratio1.compose(proba_log_ratio2);

    let alphas_betas_compo = TradeoffFunc::from_alpha_beta_vec(compo_proba_log_ratio.to_alpha_beta_vec());

    PrivacyRelation::new_fallible(move |d_in: &MI::Distance, d_out: &Vec<EpsilonDelta<MI::Distance>>| {
        if d_in.is_sign_negative() {
            return fallible!(InvalidDistance, "input sensitivity must be non-negative")
        }

        let mut result = true;
        for EpsilonDelta { epsilon, delta } in d_out {
            if epsilon.is_sign_negative() {
                return fallible!(InvalidDistance, "epsilon must be positive or 0")
            }
            if delta.is_sign_negative() {
                return fallible!(InvalidDistance, "delta must be positive or 0")
            }

            let delta_dual = alphas_betas_compo.to_epsilon_delta(epsilon.clone()).delta;
            result = result & (delta >= &delta_dual);
            if result == false {
                break;
            }
        }
        Ok(result)
    })
}

pub fn make_bounded_complexity_composition<DI, DO, MI>(
    measurement1: &Measurement<DI, DO, MI, FSmoothedMaxDivergence<MI::Distance>>,
    measurement2: &Measurement<DI, DO, MI, FSmoothedMaxDivergence<MI::Distance>>,
    npoints: u8,
    delta_min: MI::Distance,
) -> Fallible<Measurement<DI, VectorDomain<DO>, MI, FSmoothedMaxDivergence<MI::Distance>>>
    where DI: 'static + Domain,
          DO: 'static + Domain,
          MI: 'static + Metric,
          MI::Distance: 'static + MetricDistance + Clone + CastInternalReal + Tolerance + Midpoint + Float + Debug {

    let input_domain = measurement1.input_domain.clone();
    let output_domain = measurement1.output_domain.clone();
    let input_metric = measurement1.input_metric.clone();
    let output_measure = measurement2.output_measure.clone();

    if measurement2.input_domain != input_domain {
        return fallible!(DomainMismatch, "Input domains must be the same");
    }
    if measurement2.output_domain != output_domain {
        return fallible!(DomainMismatch, "Output domains must be the same");
    }
    if measurement2.input_metric != input_metric {
        return fallible!(MetricMismatch, "Input metrics must be the same");
    }
    if measurement2.output_measure != output_measure {
        return fallible!(MetricMismatch, "Output measures must be the same");
    }

    let functions = vec![measurement1.function.clone(), measurement2.function.clone()];

    Ok(Measurement::new(
        input_domain,
        VectorDomain::new(output_domain),
        Function::new_fallible(move |arg: &DI::Carrier|
            functions.iter().map(|f| f.eval(arg)).collect()),
        input_metric,
        output_measure.clone(),
        bounded_complexity_composition_privacy_relation(
            measurement1.privacy_relation.clone(),
            measurement2.privacy_relation.clone(),
            npoints,
            delta_min
        ),
    ))
}

pub fn make_bounded_complexity_composition_multi<DI, DO, MI>(
    measurements: &Vec<&Measurement<DI, DO, MI, FSmoothedMaxDivergence<MI::Distance>>>,
    npoints: u8,
    delta_min: MI::Distance,
) -> Fallible<Measurement<DI, VectorDomain<DO>, MI, FSmoothedMaxDivergence<MI::Distance>>>
    where DI: 'static + Domain,
          DO: 'static + Domain,
          MI: 'static + Metric,
          MI::Distance: 'static + MetricDistance + Clone + CastInternalReal + Tolerance + Midpoint + Float + Debug {

    if measurements.is_empty() {
        return fallible!(MakeMeasurement, "Must have at least one measurement")
    }

    let input_domain = measurements[0].input_domain.clone();
    let output_domain = measurements[0].output_domain.clone();
    let input_metric = measurements[0].input_metric.clone();
    let output_measure = measurements[0].output_measure.clone();

    if !measurements.iter().all(|v| input_domain == v.input_domain) {
        return fallible!(DomainMismatch, "All input domains must be the same");
    }
    if !measurements.iter().all(|v| output_domain == v.output_domain) {
        return fallible!(DomainMismatch, "All output domains must be the same");
    }
    if !measurements.iter().all(|v| input_metric == v.input_metric) {
        return fallible!(MetricMismatch, "All input metrics must be the same");
    }
    if !measurements.iter().all(|v| output_measure == v.output_measure) {
        return fallible!(MetricMismatch, "All output measures must be the same");
    }

    let mut functions = Vec::new();
    let mut relations = Vec::new();
    for measurement in measurements {
        functions.push(measurement.function.clone());
        relations.push(measurement.privacy_relation.clone());
    }

    let mut composed_privacy_relation = measurements[0].privacy_relation.clone();

    let size = relations.iter().len();
    for i in 1..size {
        composed_privacy_relation = bounded_complexity_composition_privacy_relation(
            composed_privacy_relation,
            relations[i].clone(),
            npoints,
            delta_min
        )
    }

    Ok(Measurement::new(
        input_domain,
        VectorDomain::new(output_domain),
        Function::new_fallible(move |arg: &DI::Carrier|
            functions.iter().map(|f| f.eval(arg)).collect()),
        input_metric,
        output_measure.clone(),
        composed_privacy_relation,
    ))
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use crate::core::*;
    use crate::dist::{L1Distance, MaxDivergence};
    use crate::dom::AllDomain;
    use crate::error::ExplainUnwrap;

    use super::*;
    use crate::meas::make_base_laplace;

    #[test]
    fn test_make_chain_mt() {
        let input_domain0 = AllDomain::<u8>::new();
        let output_domain0 = AllDomain::<i32>::new();
        let function0 = Function::new(|a: &u8| (a + 1) as i32);
        let input_metric0 = L1Distance::<i32>::default();
        let output_metric0 = L1Distance::<i32>::default();
        let stability_relation0 = StabilityRelation::new_from_constant(1);
        let transformation0 = Transformation::new(input_domain0, output_domain0, function0, input_metric0, output_metric0, stability_relation0);
        let input_domain1 = AllDomain::<i32>::new();
        let output_domain1 = AllDomain::<f64>::new();
        let function1 = Function::new(|a: &i32| (a + 1) as f64);
        let input_metric1 = L1Distance::<i32>::default();
        let output_measure1 = MaxDivergence::default();
        let privacy_relation1 = PrivacyRelation::new(|_d_in: &i32, _d_out: &f64| true);
        let measurement1 = Measurement::new(input_domain1, output_domain1, function1, input_metric1, output_measure1, privacy_relation1);
        let chain = make_chain_mt(&measurement1, &transformation0, None).unwrap_test();
        let arg = 99_u8;
        let ret = chain.function.eval(&arg).unwrap_test();
        assert_eq!(ret, 101.0);
    }

    #[test]
    fn test_make_chain_tt() {
        let input_domain0 = AllDomain::<u8>::new();
        let output_domain0 = AllDomain::<i32>::new();
        let function0 = Function::new(|a: &u8| (a + 1) as i32);
        let input_metric0 = L1Distance::<i32>::default();
        let output_metric0 = L1Distance::<i32>::default();
        let stability_relation0 = StabilityRelation::new_from_constant(1);
        let transformation0 = Transformation::new(input_domain0, output_domain0, function0, input_metric0, output_metric0, stability_relation0);
        let input_domain1 = AllDomain::<i32>::new();
        let output_domain1 = AllDomain::<f64>::new();
        let function1 = Function::new(|a: &i32| (a + 1) as f64);
        let input_metric1 = L1Distance::<i32>::default();
        let output_metric1 = L1Distance::<i32>::default();
        let stability_relation1 = StabilityRelation::new_from_constant(1);
        let transformation1 = Transformation::new(input_domain1, output_domain1, function1, input_metric1, output_metric1, stability_relation1);
        let chain = make_chain_tt(&transformation1, &transformation0, None).unwrap_test();
        let arg = 99_u8;
        let ret = chain.function.eval(&arg).unwrap_test();
        assert_eq!(ret, 101.0);
    }

    #[test]
    fn test_make_basic_composition() {
        let input_domain0 = AllDomain::<i32>::new();
        let output_domain0 = AllDomain::<f32>::new();
        let function0 = Function::new(|arg: &i32| (arg + 1) as f32);
        let input_metric0 = L1Distance::<i32>::default();
        let output_measure0 = MaxDivergence::default();
        let privacy_relation0 = PrivacyRelation::new(|_d_in: &i32, _d_out: &f64| true);
        let measurement0 = Measurement::new(input_domain0, output_domain0, function0, input_metric0, output_measure0, privacy_relation0);
        let input_domain1 = AllDomain::<i32>::new();
        let output_domain1 = AllDomain::<f64>::new();
        let function1 = Function::new(|arg: &i32| (arg - 1) as f64);
        let input_metric1 = L1Distance::<i32>::default();
        let output_measure1 = MaxDivergence::default();
        let privacy_relation1 = PrivacyRelation::new(|_d_in: &i32, _d_out: &f64| true);
        let measurement1 = Measurement::new(input_domain1, output_domain1, function1, input_metric1, output_measure1, privacy_relation1);
        let composition = make_basic_composition(&measurement0, &measurement1).unwrap_test();
        let arg = 99;
        let ret = composition.function.eval(&arg).unwrap_test();
        assert_eq!(ret, (100_f32, 98_f64));
    }

    #[test]
    fn test_make_basic_composition_multi() -> Fallible<()> {
        let measurements = vec![
            make_base_laplace::<AllDomain<_>>(0.)?,
            make_base_laplace(0.)?
        ];
        let composition = make_basic_composition_multi(&measurements.iter().collect())?;
        let arg = 99.;
        let ret = composition.function.eval(&arg)?;

        assert_eq!(ret.len(), 2);
        assert_eq!(ret, vec![99., 99.]);

        let measurements = vec![
            make_base_laplace::<AllDomain<_>>(1.)?,
            make_base_laplace(1.)?
        ];
        let composition = make_basic_composition_multi(&measurements.iter().collect())?;
        // runs once because it sits on a power of 2
        assert!(composition.privacy_relation.eval(&1., &2.)?);
        // runs a few steps- it will tighten to within TOLERANCE of 1 on the first measurement
        assert!(composition.privacy_relation.eval(&1., &2.0001)?);
        // should fail
        assert!(!composition.privacy_relation.eval(&1., &1.999)?);
        Ok(())
    }
}


impl<DI, DX, DO, MI, MX, MO> Shr<Measurement<DX, DO, MX, MO>> for Transformation<DI, DX, MI, MX>
    where DI: 'static + Domain,
          DX: 'static + Domain,
          DO: 'static + Domain,
          MI: 'static + Metric,
          MX: 'static + Metric,
          MO: 'static + Measure {
    type Output = Fallible<Measurement<DI, DO, MI, MO>>;

    fn shr(self, rhs: Measurement<DX, DO, MX, MO>) -> Self::Output {
        make_chain_mt(&rhs, &self, None)
    }
}

impl<DI, DX, DO, MI, MX, MO> Shr<Measurement<DX, DO, MX, MO>> for Fallible<Transformation<DI, DX, MI, MX>>
    where DI: 'static + Domain,
          DX: 'static + Domain,
          DO: 'static + Domain,
          MI: 'static + Metric,
          MX: 'static + Metric,
          MO: 'static + Measure {
    type Output = Fallible<Measurement<DI, DO, MI, MO>>;

    fn shr(self, rhs: Measurement<DX, DO, MX, MO>) -> Self::Output {
        make_chain_mt(&rhs, &self?, None)
    }
}

impl<DI, DX, DO, MI, MX, MO> Shr<Transformation<DX, DO, MX, MO>> for Transformation<DI, DX, MI, MX>
    where DI: 'static + Domain,
          DX: 'static + Domain,
          DO: 'static + Domain,
          MI: 'static + Metric,
          MX: 'static + Metric,
          MO: 'static + Metric {
    type Output = Fallible<Transformation<DI, DO, MI, MO>>;

    fn shr(self, rhs: Transformation<DX, DO, MX, MO>) -> Self::Output {
        make_chain_tt(&rhs, &self, None)
    }
}

impl<DI, DX, DO, MI, MX, MO> Shr<Transformation<DX, DO, MX, MO>> for Fallible<Transformation<DI, DX, MI, MX>>
    where DI: 'static + Domain,
          DX: 'static + Domain,
          DO: 'static + Domain,
          MI: 'static + Metric,
          MX: 'static + Metric,
          MO: 'static + Metric {
    type Output = Fallible<Transformation<DI, DO, MI, MO>>;

    fn shr(self, rhs: Transformation<DX, DO, MX, MO>) -> Self::Output {
        make_chain_tt(&rhs, &self?, None)
    }
}


#[cfg(test)]
mod tests_shr {
    use crate::meas::geometric::make_base_geometric;
    use crate::trans::{make_bounded_sum, make_cast_default, make_clamp, make_split_lines};

    use super::*;

    #[test]
    fn test_shr() -> Fallible<()> {
        (
            make_split_lines()? >>
            make_cast_default()? >>
            make_clamp(0, 1)? >>
            make_bounded_sum(0, 1)? >>
            make_base_geometric(1., Some((0, 10)))?
        ).map(|_| ())
    }
}
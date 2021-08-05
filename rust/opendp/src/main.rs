//use opendp::meas::gaussian::*;
//use opendp::dom::{AllDomain};//, VectorDomain};
use opendp::dist::{FSmoothedMaxDivergence, MaxDivergence, EpsilonDelta};
use opendp::dom::AllDomain;
use opendp::chain::*;
use opendp::meas::{make_base_laplace, make_base_gaussian};

//use opendp::samplers::CastInternalReal;


fn main(){
    // Laplace with FSmoothedMaxDivergence
    println!("\nLaplace with FSmoothedMaxDivergence");
    let scale1 = 1.0;
    let measurement1 = make_base_laplace::<AllDomain<_>, FSmoothedMaxDivergence<_>>(scale1).unwrap();
    let predicate1 = measurement1.privacy_relation.clone();
    let d_in = 1.0;
    let d_out = vec![
        EpsilonDelta { epsilon: 1., delta: 0.0 },
        EpsilonDelta { epsilon: 0.75, delta: 0.12 },
        EpsilonDelta { epsilon: 0.5, delta: 0.23 },
        EpsilonDelta { epsilon: 0.25, delta: 0.32 },
        EpsilonDelta { epsilon: 0., delta: 0.4 }
    ];

    println!("privacy_relation: {}", predicate1.eval(&d_in, &d_out).unwrap());
    println!("\n Find epsilon");
    let deltas = vec![0.0, 1. / 1000., 1. / 100., 1. / 10.];
    for delta in deltas {
        println!("for delta = {}, epsilon = {:#?}", delta, predicate1.find_epsilon(&d_in, delta).unwrap().unwrap());
    }
    println!("\n Find delta");
    let epsilons = vec![1.2, 1., 0.8, 0.5, 0.2];
    for epsilon in epsilons {
        println!("for epsilon = {}, delta = {:#?}", epsilon, predicate1.find_delta(&d_in, epsilon).unwrap().unwrap());
    }

    let scale2 = 1. / 1.5;
    let measurement2 = make_base_laplace::<AllDomain<_>, FSmoothedMaxDivergence<_>>(scale2).unwrap();
    let predicate2 = measurement2.privacy_relation.clone();
    println!("\n Find epsilon");
    let deltas = vec![0.0, 1. / 1000., 1. / 100., 1. / 10.];
    for delta in deltas {
        println!("for delta = {}, epsilon = {:#?}", delta, predicate2.find_epsilon(&d_in, delta).unwrap().unwrap());
    }
    println!("\n Find delta");
    let epsilons = vec![1.2, 1., 0.8, 0.5, 0.2];
    for epsilon in epsilons {
        println!("for epsilon = {}, delta = {:#?}", epsilon, predicate2.find_delta(&d_in, epsilon).unwrap().unwrap());
    }

    // Composition
    let npoints: u8 = 9;
    let delta_min = 0.00000000001;
    let measurements = vec![measurement1, measurement2];
    let composition = make_bounded_complexity_composition_multi(
        &measurements.iter().collect(),
        npoints,
        delta_min,
    ).unwrap();

    // Laplace with FSmoothedMaxDivergence
    // println!("\nLaplace with FSmoothedMaxDivergence");
    // let scale = 1.0;
    // let measurement = make_base_laplace::<AllDomain<_>, FSmoothedMaxDivergence<_>>(scale).unwrap();
    // let predicate = measurement.privacy_relation;
    // let d_in = 1.0;
    // let d_out = vec![
    //     EpsilonDelta { epsilon: 1., delta: 0.0 },
    //     EpsilonDelta { epsilon: 0.75, delta: 0.12 },
    //     EpsilonDelta { epsilon: 0.5, delta: 0.23 },
    //     EpsilonDelta { epsilon: 0.25, delta: 0.32 },
    //     EpsilonDelta { epsilon: 0., delta: 0.4 }
    // ];
    // println!("privacy_relation: {}", predicate.eval(&d_in, &d_out).unwrap());
    // let delta = 0.23;
    // println!("delta = {}, find epsilon: {:#?}", delta, predicate.find_epsilon(&d_in, delta).unwrap().unwrap());
    // let epsilon = 0.;
    // println!("epsilon = {}, find delta: {:#?}", epsilon, predicate.find_delta(&d_in, epsilon).unwrap().unwrap());

    // let epsilons: Vec<f32> = vec![0.1, 0.2, 0.5, 1.];
    // let epsilon_deltas: Vec<EpsilonDelta<f32>> = epsilons.iter()
    //     .map(|eps| EpsilonDelta{ epsilon: *eps, delta: predicate.find_delta(&d_in, *eps).unwrap().unwrap()})
    //     .collect();
    // println!("epsilons_deltas = {:#?}", epsilon_deltas);

    // // Compute alpha beta
    // let alphas_betas = compute_alpha_beta(&epsilon_deltas);
    // println!("alphas_betas = {:#?}", alphas_betas);
    // // Compute probas
    // let probas = compute_adjacent_probabilities(&alphas_betas);
    // println!("probas = {:#?}", probas);


    // // Gaussian with FSmoothedMaxDivergence
    // println!("\nGaussian with FSmoothedMaxDivergence");
    // let scale = 1.0;
    // let measurement = make_base_gaussian::<AllDomain<_>, FSmoothedMaxDivergence<_>>(scale).unwrap();
    // let predicate = measurement.privacy_relation.clone();
    // let d_in = 1.0;
    // let d_out = vec![
    //     EpsilonDelta { epsilon: 0.25, delta: 0.32 },
    //     EpsilonDelta { epsilon: 0.5, delta: 0.23 },
    //     EpsilonDelta { epsilon: 0.75, delta: 0.12 },
    // ];
    // println!("privacy_relation: {}", predicate.eval(&d_in, &d_out).unwrap());
    // //let delta = 0.23;
    // //println!("delta = {}, find epsilon: {:#?}", delta, predicate.find_epsilon(&d_in, delta).unwrap().unwrap());
    // //let epsilon = 0.;
    // //println!("epsilon = {}, find delta: {:#?}", epsilon, predicate.find_delta(&d_in, epsilon).unwrap().unwrap());
    // let relations = vec![measurement.privacy_relation.clone(), measurement.privacy_relation.clone()];

    // let compo = bounded_complexity_composition(&relations, &d_in, &d_out);
    // println!("compo = {:?}", compo);

    if false {
        // make_basic_composition_multi
        let meas1 = make_base_laplace::<AllDomain<_>, MaxDivergence<_>>(1.).unwrap();
        let meas2 = make_base_laplace::<AllDomain<_>, MaxDivergence<_>>(1.).unwrap();
        println!("1.");
        let measurements = vec![meas1, meas2];
        let composition = make_basic_composition_multi(&measurements.iter().collect()).unwrap();
        let pr1 =  composition.privacy_relation.eval(&1., &2.).unwrap();
        println!("{}",pr1);
        let pr2 =  composition.privacy_relation.eval(&1., &2.0001).unwrap();
        println!("{}",pr2);
        let pr3 = composition.privacy_relation.eval(&1., &1.999).unwrap(); // should fail
        println!("{}",pr3);
    }
}
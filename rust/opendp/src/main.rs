use opendp::dist::{FSmoothedMaxDivergence, MaxDivergence, EpsilonDelta};
use opendp::dom::AllDomain;
use opendp::chain::*;
use opendp::meas::{make_base_laplace, make_base_gaussian};



fn main(){
    // let npoints: u8 = 9;
    // let delta_min = 0.0001;

    // // Laplace with FSmoothedMaxDivergence
    // let scale1:f64 = 1.0;
    // let measurement1 = make_base_gaussian::<AllDomain<f64>, FSmoothedMaxDivergence<f64>>(scale1).unwrap();
    // let predicate1 = measurement1.privacy_relation.clone();
    // let d_in = 1.0;
    // let d_out = vec![
    //     EpsilonDelta { epsilon: 0.75, delta: 0.12 },
    //     EpsilonDelta { epsilon: 0.5, delta: 0.23 },
    //     EpsilonDelta { epsilon: 0.25, delta: 0.32 },
    // ];

    // println!("privacy_relation: {}", predicate1.eval(&d_in, &d_out).unwrap());
    // println!("\n Find epsilon");
    // let deltas = vec![1. / 1000., 1. / 100., 1. / 10.];
    // for delta in deltas {
    //     println!("for delta = {}, epsilon = {:#?}", delta, predicate1.find_epsilon(&d_in, delta).unwrap());
    // }
    // println!("\n Find delta");
    // let epsilons = vec![1.2, 1., 0.8];
    // for epsilon in epsilons {
    //     println!("for epsilon = {}, delta = {:#?}", epsilon, predicate1.find_delta(&d_in, epsilon).unwrap());
    // }

    // let eps_delta_family = predicate1.find_epsilon_delta_family(
    //     npoints.clone(),
    //     delta_min.clone(),
    //     d_in
    // );
    // println!("\n1. eps_delta_family: {:#?}", eps_delta_family);

    // let scale2:f64 = 1. / 1.5;
    // let measurement2 = make_base_gaussian::<AllDomain<f64>, FSmoothedMaxDivergence<f64>>(scale2).unwrap();
    // let predicate2 = measurement2.privacy_relation.clone();
    // let eps_delta_family = predicate2.find_epsilon_delta_family(
    //     npoints.clone(),
    //     delta_min.clone(),
    //     d_in
    // );
    // println!("\n2. eps_delta_family: {:#?}", eps_delta_family);

    // println!("\n Find epsilon");
    // let deltas = vec![1. / 1000., 1. / 100., 1. / 10.];
    // for delta in deltas {
    //     println!("for delta = {}, epsilon = {:#?}", delta, predicate2.find_epsilon(&d_in, delta).unwrap());
    // }
    // println!("\n Find delta");
    // let epsilons = vec![1.2, 1., 0.8, 0.5, 0.2];
    // for epsilon in epsilons {
    //     println!("for epsilon = {}, delta = {:#?}", epsilon, predicate2.find_delta(&d_in, epsilon).unwrap());
    // }

    // //Composition
    // let composition = make_bounded_complexity_composition(
    //     &measurement1,
    //     &measurement2,
    //     npoints,
    //     delta_min,
    // ).unwrap();
    // let d_out = vec![
    //     EpsilonDelta { epsilon: 3., delta: delta_min.clone()},
    // ];
    // println!("\n\nprivacy_relation: {}", composition.privacy_relation.eval(&d_in, &d_out).unwrap());
    // let eps_delta_family = composition.privacy_relation.find_epsilon_delta_family(
    //     npoints.clone(),
    //     delta_min.clone(),
    //     d_in
    // );
    // println!("\neps_delta_family: {:#?}", eps_delta_family);

    // // Composition with multi
    // let measurements = vec![&measurement1, &measurement2];
    // let composition = make_bounded_complexity_composition_multi(
    //     &measurements,
    //     npoints,
    //     delta_min,
    // ).unwrap();
    // let d_out = vec![
    //     EpsilonDelta { epsilon: 3., delta: delta_min.clone()},
    // ];
    // println!("\n\nprivacy_relation: {}", composition.privacy_relation.eval(&d_in, &d_out).unwrap());
    // let eps_delta_family = composition.privacy_relation.find_epsilon_delta_family(
    //     npoints.clone(),
    //     delta_min.clone(),
    //     d_in
    // );
    // println!("\neps_delta_family: {:#?}", eps_delta_family);

    println!("\n================\nTests\n================\n");
    let npoints: u8 = 5;
    let delta_min = 1e-5;
    let measurements = vec![
            make_base_laplace::<AllDomain<_>, FSmoothedMaxDivergence<_>>(0.).unwrap(),
            make_base_laplace::<AllDomain<_>, FSmoothedMaxDivergence<_>>(0.).unwrap()
        ];
        let composition = make_bounded_complexity_composition_multi(
            &measurements.iter().collect(),
            npoints,
            delta_min,
        ).unwrap();
        let d_out = vec![
            EpsilonDelta { epsilon: 2., delta: delta_min.clone()},
        ];
        println!("{}", composition.privacy_relation.eval(&1., &d_out).unwrap());
        let d_out = vec![
            EpsilonDelta { epsilon: 2.0001, delta: delta_min.clone()},
        ];
        println!("{}", composition.privacy_relation.eval(&1., &d_out).unwrap());
        let d_out = vec![
            EpsilonDelta { epsilon: 1.5, delta: delta_min.clone()},
        ];
        println!("{}", composition.privacy_relation.eval(&1., &d_out).unwrap());

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
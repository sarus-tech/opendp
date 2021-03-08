import opendp

def main():
    lib_path = "../rust/target/debug/libopendp_ffi.dylib"
    odp = opendp.OpenDP(lib_path)

    ### HELLO WORLD
    identity = odp.trans.make_identity(b"<String>")
    arg = odp.data.from_string(b"hello, world!")
    ret = odp.core.transformation_invoke(identity, arg)
    print(odp.to_str(ret))
    odp.data.data_free(arg)
    odp.data.data_free(ret)
    odp.core.transformation_free(identity)

    ### SUMMARY STATS
    # Parse dataframe
    split_dataframe = odp.trans.make_split_dataframe(b",", 3)
    parse_column_1 = odp.trans.make_parse_column(b"<i32>", b"1", True)
    parse_column_2 = odp.trans.make_parse_column(b"<f64>", b"2", True)
    parse_dataframe = odp.make_chain_tt_multi(parse_column_2, parse_column_1, split_dataframe)

    # Noisy sum, col 1
    select_1 = odp.trans.make_select_column(b"<i32>", b"1")
    clamp_1 = odp.trans.make_clamp(b"<i32>", odp.i32_p(0), odp.i32_p(10))
    bounded_sum_1 = odp.trans.make_bounded_sum_l1(b"<i32>", odp.i32_p(0), odp.i32_p(10))
    base_laplace_1 = odp.meas.make_base_laplace(b"<i32>", 1.0)
    noisy_sum_laplace = odp.core.make_chain_mt(base_laplace_1, odp.make_chain_tt_multi(bounded_sum_1, clamp_1, select_1))

    # Count, col 2
    select_2 = odp.trans.make_select_column(b"<f64>", b"2")
    count_2 = odp.trans.make_count_l2(b"<f64>")
    base_laplace_2 = odp.meas.make_base_laplace(b"<u32>", 1.0)
    noisy_count_laplace = odp.core.make_chain_mt(base_laplace_2, odp.make_chain_tt_multi(count_2, select_2))

    # Compose & chain
    composition = odp.core.make_composition(noisy_sum_laplace, noisy_count_laplace)
    everything = odp.core.make_chain_tt(composition, parse_dataframe)

    # Do it!!!
    arg = odp.data.from_string(b"ant, 1, 1.1\nbat, 2, 2.2\ncat, 3, 3.3")
    ret = odp.core.measurement_invoke(everything, arg)
    print(odp.to_str(ret))

    # Clean up
    odp.data.data_free(arg)
    odp.data.data_free(ret)
    odp.core.measurement_free(everything)

    split_dataframe = odp.trans.make_split_dataframe(b",", 3)
    parse_column_1 = odp.trans.make_parse_column(b"<i32>", b"1", True)
    parse_column_2 = odp.trans.make_parse_column(b"<f64>", b"2", True)
    parse_dataframe = odp.make_chain_tt_multi(parse_column_2, parse_column_1, split_dataframe)

    select_3 = odp.trans.make_select_column(b"<i32>", b"1")
    clamp_3 = odp.trans.make_clamp(b"<i32>", odp.i32_p(0), odp.i32_p(10))
    bounded_sum_3 = odp.trans.make_bounded_sum_l2(b"<i32>", odp.i32_p(0), odp.i32_p(10))
    base_gaussian_1 = odp.meas.make_base_gaussian(b"<i32>", 1.0)
    noisy_sum_gaussian = odp.core.make_chain_mt(base_gaussian_1, odp.make_chain_tt_multi(bounded_sum_3, clamp_3, select_3))

    select_4 = odp.trans.make_select_column(b"<f64>", b"2")
    count_4 = odp.trans.make_count_l1(b"<f64>")
    base_gaussian = odp.meas.make_base_gaussian(b"<u32>",1)
    noisy_count_gaussian = odp.core.make_chain_mt(base_gaussian, odp.make_chain_tt_multi(count_4, select_4))
    composition = odp.core.make_composition(noisy_sum_gaussian, noisy_count_gaussian)
    everything = odp.core.make_chain_tt(composition, parse_dataframe)

    arg = odp.data.from_string(b"ant, 1, 1.1\nbat, 2, 2.2\ncat, 3, 3.3")
    ret = odp.core.measurement_invoke(everything, arg )
    print(odp.to_str(ret))

    # Clean up
    odp.data.data_free(arg)
    odp.data.data_free(ret)
    odp.core.measurement_free(everything)

    split_dataframe = odp.trans.make_split_dataframe(b",", 3)
    parse_column_1 = odp.trans.make_parse_column(b"<i32>", b"1", True)
    parse_column_2 = odp.trans.make_parse_column(b"<f64>", b"2", True)
    parse_dataframe = odp.make_chain_tt_multi(parse_column_2, parse_column_1, split_dataframe)

    select_5 = odp.trans.make_select_column(b"<i32>", b"1")
    base_laplace_vec_1 = odp.meas.make_base_laplace_vec(b"<i32>", 1.0)
    noisy_records_laplace_vec_1 = odp.core.make_chain_mt( base_laplace_vec_1, select_5)

    clamp_4 = odp.trans.make_clamp(b"<i32>", odp.i32_p(0), odp.i32_p(10))
    bounded_sum_4 = odp.trans.make_bounded_sum_l2(b"<i32>", odp.i32_p(0), odp.i32_p(10))
    noisy_sum_4 = odp.make_chain_tt_multi(bounded_sum_4, clamp_4, noisy_records_laplace_vec_1)

    select_6 = odp.trans.make_select_column(b"<f64>", b"2")
    count_5 = odp.trans.make_count_l1(b"<f64>")
    noisy_count_5 = odp.make_chain_tt_multi(count_5, select_6)
    composition = odp.core.make_composition(noisy_sum_4, noisy_count_5)
    everything = odp.core.make_chain_tt(composition, parse_dataframe)

    arg = odp.data.from_string(b"ant, 1, 1.1\nbat, 2, 2.2\ncat, 3, 3.3")
    ret = odp.core.measurement_invoke(everything, arg)
    print(odp.to_str(ret))

    # Clean up
    odp.data.data_free(arg)
    odp.data.data_free(ret)
    odp.core.measurement_free(everything)


if __name__ == "__main__":
    main()

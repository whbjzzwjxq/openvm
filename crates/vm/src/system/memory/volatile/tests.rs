use std::{borrow::BorrowMut, collections::HashSet, iter, sync::Arc};

use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
use openvm_stark_backend::{
    interaction::BusIndex,
    p3_field::FieldAlgebra,
    p3_matrix::dense::RowMajorMatrix,
    prover::{cpu::CpuBackend, types::AirProvingContext},
    AirRef, Chip, ChipUsageGetter,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
    utils::create_seeded_rng,
};
use rand::Rng;
use test_log::test;

use crate::system::memory::{
    offline_checker::MemoryBus, volatile::VolatileBoundaryChip, TimestampedEquipartition,
    TimestampedValues,
};

type Val = BabyBear;

#[test]
fn boundary_air_test() {
    let mut rng = create_seeded_rng();

    const MEMORY_BUS: BusIndex = 1;
    const RANGE_CHECKER_BUS: BusIndex = 3;
    const MAX_ADDRESS_SPACE: u32 = 4;
    const LIMB_BITS: usize = 15;
    const MAX_VAL: u32 = 1 << LIMB_BITS;
    const DECOMP: usize = 8;
    let memory_bus = MemoryBus::new(MEMORY_BUS);

    let num_addresses = 10;
    let mut distinct_addresses = HashSet::new();
    while distinct_addresses.len() < num_addresses {
        let addr_space = rng.gen_range(0..MAX_ADDRESS_SPACE);
        let pointer = rng.gen_range(0..MAX_VAL);
        distinct_addresses.insert((addr_space, pointer));
    }

    let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, DECOMP);
    let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));
    let mut boundary_chip =
        VolatileBoundaryChip::new(memory_bus, 2, LIMB_BITS, range_checker.clone());

    let mut final_memory = TimestampedEquipartition::new();

    for (addr_space, pointer) in distinct_addresses.iter().cloned() {
        let final_data = Val::from_canonical_u32(rng.gen_range(0..MAX_VAL));
        let final_clk = rng.gen_range(1..MAX_VAL) as u32;

        final_memory.push((
            (addr_space, pointer),
            TimestampedValues {
                values: [final_data],
                timestamp: final_clk,
            },
        ));
    }
    final_memory.sort_by_key(|(key, _)| *key);

    let diff_height = num_addresses.next_power_of_two() - num_addresses;

    let init_memory_dummy_air = DummyInteractionAir::new(4, false, MEMORY_BUS);
    let final_memory_dummy_air = DummyInteractionAir::new(4, true, MEMORY_BUS);

    let init_memory_trace = Arc::new(RowMajorMatrix::new(
        distinct_addresses
            .iter()
            .flat_map(|(addr_space, pointer)| {
                vec![
                    Val::ONE,
                    Val::from_canonical_u32(*addr_space),
                    Val::from_canonical_u32(*pointer),
                    Val::ZERO,
                    Val::ZERO,
                ]
            })
            .chain(iter::repeat_n(Val::ZERO, 5 * diff_height))
            .collect(),
        5,
    ));

    let final_memory_trace = Arc::new(RowMajorMatrix::new(
        distinct_addresses
            .iter()
            .flat_map(|(addr_space, pointer)| {
                let timestamped_value = final_memory[final_memory
                    .binary_search_by(|(key, _)| key.cmp(&(*addr_space, *pointer)))
                    .unwrap()]
                .1;

                vec![
                    Val::ONE,
                    Val::from_canonical_u32(*addr_space),
                    Val::from_canonical_u32(*pointer),
                    timestamped_value.values[0],
                    Val::from_canonical_u32(timestamped_value.timestamp),
                ]
            })
            .chain(iter::repeat_n(Val::ZERO, 5 * diff_height))
            .collect(),
        5,
    ));

    boundary_chip.finalize(final_memory.clone());
    let boundary_air = Arc::new(boundary_chip.air.clone()) as AirRef<_>;
    let boundary_ctx: AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> =
        boundary_chip.generate_proving_ctx(());
    // test trace height override
    {
        let overridden_height = boundary_ctx.main_trace_height() * 2;
        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let mut boundary_chip =
            VolatileBoundaryChip::new(memory_bus, 2, LIMB_BITS, range_checker.clone());
        boundary_chip.set_overridden_height(overridden_height);
        boundary_chip.finalize(final_memory.clone());
        let boundary_ctx: AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> =
            boundary_chip.generate_proving_ctx(());
        assert_eq!(
            boundary_ctx.main_trace_height(),
            overridden_height.next_power_of_two()
        );
    }

    BabyBearPoseidon2Engine::run_test_fast(
        vec![
            boundary_air,
            Arc::new(range_checker.air),
            Arc::new(init_memory_dummy_air),
            Arc::new(final_memory_dummy_air),
        ],
        vec![
            boundary_ctx,
            range_checker.generate_proving_ctx(()),
            AirProvingContext::simple_no_pis(init_memory_trace),
            AirProvingContext::simple_no_pis(final_memory_trace),
        ],
    )
    .expect("Verification failed");
}

#[test]
fn volatile_initial_data_soundness_poc() {
    use crate::system::memory::volatile::VolatileBoundaryCols;

    let mut _rng = create_seeded_rng();

    const MEMORY_BUS: BusIndex = 1;
    const RANGE_CHECKER_BUS: BusIndex = 3;
    const LIMB_BITS: usize = 15;
    const DECOMP: usize = 8;
    let memory_bus = MemoryBus::new(MEMORY_BUS);

    let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, DECOMP);
    let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));
    let mut boundary_chip =
        VolatileBoundaryChip::new(memory_bus, 2, LIMB_BITS, range_checker.clone());

    // A malicious prover wants to initialize address (1, 100) with value 0x12345678
    let mal_addr_space = 1;
    let mal_pointer = 100;
    let mal_initial_value = Val::from_canonical_u32(0x12345678);

    let mut final_memory = TimestampedEquipartition::new();
    final_memory.push((
        (mal_addr_space, mal_pointer),
        TimestampedValues {
            values: [mal_initial_value],
            timestamp: 10,
        },
    ));

    boundary_chip.finalize(final_memory.clone());

    // Generate honest proving context
    let mut boundary_ctx = boundary_chip.generate_proving_ctx(());

    // MALICIOUS ACT: Modify the trace to inject non-zero initial data
    let width = boundary_chip.trace_width();
    let main_trace = boundary_ctx.common_main.as_ref().unwrap();
    let mut rows = main_trace.values.clone();
    {
        let row_slice = &mut rows[0..width];
        let row: &mut VolatileBoundaryCols<Val> = row_slice.borrow_mut();

        // This is the bug: initial_data is NOT constrained to be zero.
        row.initial_data = mal_initial_value;
    }
    boundary_ctx.common_main = Some(Arc::new(RowMajorMatrix::new(rows, width)));

    // Mock the memory bus interactions to match the malicious initial value
    let init_memory_dummy_air = DummyInteractionAir::new(4, false, MEMORY_BUS);
    let init_memory_trace = Arc::new(RowMajorMatrix::new(
        vec![
            Val::ONE,
            Val::from_canonical_u32(mal_addr_space),
            Val::from_canonical_u32(mal_pointer),
            mal_initial_value, // Matches the malicious non-zero initial value
            Val::ZERO,
        ]
        .into_iter()
        .chain(iter::repeat_n(
            Val::ZERO,
            5 * (boundary_ctx.main_trace_height() - 1),
        ))
        .collect(),
        5,
    ));

    let final_memory_dummy_air = DummyInteractionAir::new(4, true, MEMORY_BUS);
    let final_memory_trace = Arc::new(RowMajorMatrix::new(
        vec![
            Val::ONE,
            Val::from_canonical_u32(mal_addr_space),
            Val::from_canonical_u32(mal_pointer),
            mal_initial_value,
            Val::from_canonical_u32(10),
        ]
        .into_iter()
        .chain(iter::repeat_n(
            Val::ZERO,
            5 * (boundary_ctx.main_trace_height() - 1),
        ))
        .collect(),
        5,
    ));

    // This SHOULD FAIL if there were a zero-init constraint.
    // It passes, proving the soundness bug.
    BabyBearPoseidon2Engine::run_test_fast(
        vec![
            Arc::new(boundary_chip.air.clone()),
            Arc::new(range_checker.air.clone()),
            Arc::new(init_memory_dummy_air),
            Arc::new(final_memory_dummy_air),
        ],
        vec![
            boundary_ctx,
            range_checker.generate_proving_ctx(()),
            AirProvingContext::simple_no_pis(init_memory_trace),
            AirProvingContext::simple_no_pis(final_memory_trace),
        ],
    )
    .expect("PoC failed: Malicious non-zero initialization was caught!");
}

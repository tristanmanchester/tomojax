# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 GPU memory regression isolation
- Goal: isolate and fix the 64^3/64-view v2 JAX reference memory blow-up before
  continuing realistic setup-global GPU benchmarks.

### Scope

- In scope:
  - Run setup-global 64^3 probes on GPU for view counts 1/4/16/64.
  - Isolate projector/backprojector, FISTA, Schur residual/Jacobian reductions,
    nuisance fitting, fixed-truth, stopped-reconstruction, Schur-disabled, and
    reconstruction-disabled paths.
  - Implement chunked view/parameter accumulation for the offending path.
  - Preserve correctness tests.
  - Rerun the 64^3/64-view fixed-truth and stopped-reconstruction benchmark on
    GPU after the memory fix.
- Out of scope:
  - Shrinking the realistic benchmark as a final answer.
  - Nuisance-applied datasets for this first pass.
  - New report fields unless required for GPU device/provenance.
  - Geometry tolerance relaxation.
  - Solver tuning.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.datasets`, `tomojax.align`, and `tomojax.bench`
  existing benchmark surfaces.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Verify JAX GPU device selection.
- [x] Generate nuisance-free 64^3/64-view setup-global sidecar dataset.
- [x] Run component memory probes for views 1/4/16/64.
- [x] Identify the allocation source.
- [x] Implement chunked accumulation for the offending path.
- [x] Preserve focused correctness tests and `just imports`.
- [x] Rerun 64^3/64-view fixed-truth and stopped-reconstruction GPU benchmarks.
- [x] Record pass/fail, recovery, Schur acceptance, residuals, NMSE, timing,
  GPU device, and memory-source diagnosis.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the benchmark summary slice.

### Validation

- `JAX_PLATFORMS=cuda` with the venv NVIDIA wheel library paths selected
  `jax_default_backend = "gpu"` and `cuda:0`.
- `synth128_setup_global_tomo` 64^3/64-view clean sidecar generation passed
  consistency checks.
- Initial 64^3/64-view fixed-truth run failed in Schur finite differences with
  a 12.14 GiB GPU allocation for `f32[194,64,64,64,64]`.
- Component probes for views 1/4/16/64 passed after sequential FD column
  accumulation.
- Both 64^3/64-view benchmark modes completed on GPU and wrote
  `benchmark_result.json`.
- `tomojax-synthetic-benchmark-compare` rendered
  `.artifacts/phase8_setup_global_gpu_ladder/benchmark_comparison_64.md`.
- `uv run ruff format ...` passed for touched source/tests.
- `uv run ruff check ...` passed for touched source/tests.
- `uv run basedpyright ...` passed for touched source/tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_lm_numerics.py
  tests/test_synthetic_datasets.py tests/test_joint_schur_lm.py
  tests/test_align_auto_cli.py -q` passed: 29 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Treat the 12 GiB XLA allocation as a reference-path memory regression.
- Keep benchmark artifacts on the existing sidecar/result/report/compare path
  after the memory fix.
- Fixed-truth failed at 64^3/64 views, so do not proceed to 128^3 until
  setup/pose/theta coupling or geometry convention mapping is diagnosed.

### Risks

- Risk: broad `just check` still exposes legacy Ruff debt outside this slice.
- Mitigation: run focused validation plus `just imports`, and avoid legacy Ruff
  cleanup unless explicitly requested.

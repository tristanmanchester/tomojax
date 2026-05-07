# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 supported-only oracle benchmark
- Goal: prove the 64^3/64-view setup-global supported-DOF oracle path after
  the nominal-theta fix, before judging the five-case benchmark suite.

### Scope

- In scope:
  - Generate a clean supported-only `synth128_setup_global_tomo` 64^3/64-view
    sidecar variant with nominal theta 0..180, active det_u/theta_offset,
    optional det_v, and no unsupported roll/axis/pose/nuisance/object motion.
  - Add a pose-frozen geometry-update path for the oracle run using
    fixed_synthetic_truth through the existing sidecar/align-auto artifact path.
  - Run the GPU benchmark and record benchmark_result/benchmark_report/compare
    artifacts plus a concise markdown summary.
  - Keep memory-safe sequential FD and record JAX backend/device provenance.
- Out of scope:
  - Five-case benchmark judgement.
  - Stopped-reconstruction diagnosis unless fixed-truth supported-only passes.
  - Unsupported DOF implementation.
  - New report fields unless required for this oracle artifact.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`, `tomojax.cli`, and `tomojax.datasets`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add supported-only sidecar generation/ingestion option.
- [x] Add pose-frozen fixed-truth geometry-update path.
- [x] Add focused tests for supported-only sidecar and pose-frozen CLI/config.
- [x] Run focused validation and `just imports`.
- [x] Run GPU 64^3/64-view supported-only fixed-truth benchmark.
- [x] Write benchmark summary and implementation-log entry.
- [x] Commit the supported-only oracle slice.

### Validation

- `uv run ruff check ...` on touched source/test files passed.
- `uv run basedpyright ...` on touched source/test files passed with
  0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py
  tests/test_synthetic_datasets.py tests/test_align_auto_cli.py
  tests/test_bench_synthetic_results.py -q`
  passed: 37 tests.
- `just imports` passed.
- GPU oracle command passed with JAX backend `gpu` and selected device `cuda:0`:
  `tomojax-align-auto-smoke --profile balanced --synthetic-dataset-dir
  .artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only
  --geometry-update-volume-source fixed_synthetic_truth --geometry-update-pose-frozen`.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Treat the existing five-case benchmark as blocked until this supported-only
  oracle is either passing or has a sharper fixed-truth blocker.
- The initial balanced oracle failed because Schur predicted reduction was on
  raw residual-sum scale while actual reduction was mean loss; scaling by data
  rows made trust adaptation comparable and the oracle passed.

### Risks

- Risk: broad `just check` still exposes legacy Ruff debt outside this slice.
- Mitigation: run focused validation plus `just imports`, and avoid legacy Ruff
  cleanup unless explicitly requested.

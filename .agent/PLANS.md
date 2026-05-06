# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: extend the deterministic 32^3 benchmark pass to all five planned cases.

### Scope

- In scope:
  - Generate the remaining planned synthetic128 32^3 sidecar dataset.
  - Run `align-auto` on the existing generated sidecar directory.
  - Refresh the collected `benchmark_result.json` comparison markdown.
  - Record pass/fail, timing, and recovery summary in
    `docs/implementation_log.md`.
- Out of scope:
  - Adding or changing artifact/report/observability fields.
  - New benchmark ingestion behavior.
  - Solver tuning beyond command-line flags already supported.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` CLI run artifacts with public
  `tomojax.datasets` sidecar generation/loading and `tomojax.bench` comparison.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Generate the remaining deterministic 32^3 sidecar dataset.
- [x] Run `tomojax-align-auto-smoke` on the existing sidecar directory.
- [x] Refresh all-five benchmark result comparison.
- [x] Record all-five pass/fail, timing, and recovery summary.
- [x] Commit the all-five benchmark summary.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run python` generated
  `synth128_combined_nuisance_jumps_32` through public
  `tomojax.datasets.generate_synthetic_dataset`.
- `JAX_PLATFORM_NAME=cpu uv run tomojax-align-auto-smoke ...` completed for the
  existing `synth128_combined_nuisance_jumps_32` sidecar directory.
- `uv run tomojax-synthetic-benchmark-compare ... --out .artifacts/phase8_multi_case_32/benchmark_comparison.md`
  passed for all five result artifacts.
- `just imports` passed after recording the all-five documentation summary.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Use generated local artifacts under ignored `.artifacts/`; commit a concise
  documentation summary unless a small report artifact is suitable.

### Risks

- Risk: 32^3 smoke profiles may fail planned synthetic128 pass criteria while
  still producing useful ingestion evidence.
- Mitigation: record exact pass/fail labels, timings, and recovery metrics from
  `benchmark_result.json`.

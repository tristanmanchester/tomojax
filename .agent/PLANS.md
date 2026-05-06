# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: run the first deterministic multi-case 32^3 synthetic benchmark pass.

### Scope

- In scope:
  - Generate 3-5 planned synthetic128 scenario sidecar datasets at 32^3.
  - Run `align-auto` on each existing generated sidecar directory.
  - Collect `benchmark_result.json` files and render the compare CLI markdown.
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

- [x] Generate 3-5 deterministic 32^3 sidecar datasets.
- [x] Run `tomojax-align-auto-smoke` on each existing sidecar directory.
- [x] Collect benchmark result artifacts and render the comparison report.
- [x] Record benchmark pass/fail, timing, and recovery summary.
- [x] Commit the benchmark summary or intended artifacts.

### Validation

- `uv run python` generated four 32^3 sidecar datasets through public
  `tomojax.datasets.generate_synthetic_dataset`.
- `JAX_PLATFORM_NAME=cpu uv run tomojax-align-auto-smoke ...` completed for all
  four existing sidecar directories.
- `uv run tomojax-synthetic-benchmark-compare ... --out .artifacts/phase8_multi_case_32/benchmark_comparison.md`
  passed.
- `just imports` passed after recording the documentation summary.

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

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: run the first deterministic multi-case 32^3 synthetic benchmark pass
  through existing sidecar directories and compare artifacts.

### Scope

- In scope:
  - Generate at least 3-5 deterministic 32^3 sidecar datasets from planned
    synthetic benchmark scenarios.
  - Run `tomojax-align-auto-smoke` on each existing sidecar directory.
  - Collect `benchmark_result.json` artifacts and render a comparison markdown
    report with the compare CLI.
  - Record pass/fail, timing, and recovery outcomes in
    `docs/implementation_log.md`.
  - Commit either the benchmark artifacts or a concise benchmark summary.
- Out of scope:
  - Adding artifact/report/observability fields.
  - More artifact-shape polishing.
  - New benchmark-ingestion behavior beyond using existing sidecars.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`, `tomojax.datasets`, and `tomojax.bench`
  public benchmark/reporting surfaces.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Inspect dataset generation and compare CLI contracts.
- [x] Generate or confirm 3-5 deterministic 32^3 sidecar datasets.
- [x] Run `tomojax-align-auto-smoke` against each sidecar directory.
- [x] Render the benchmark comparison report.
- [x] Record outcomes in `docs/implementation_log.md`.
- [x] Run focused validation and `just imports`.
- [x] Commit the benchmark summary/artifacts slice.

### Validation

- `uv run python` generated five 32^3 sidecar directories under
  `.artifacts/phase8_multi_case_32_benchmark_pass/datasets`.
- Five `JAX_PLATFORM_NAME=cpu uv run tomojax-align-auto-smoke ...` runs
  completed under `.artifacts/phase8_multi_case_32_benchmark_pass/runs`.
- `JAX_PLATFORM_NAME=cpu uv run tomojax-synthetic-benchmark-compare ... --out
  .artifacts/phase8_multi_case_32_benchmark_pass/benchmark_comparison.md`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_bench_synthetic_results.py -q`
  passed: 6 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Commit the concise markdown benchmark summary rather than ignored generated
  `.npy` arrays and run directories.
- This pass intentionally records the current stopped-reconstruction failures
  instead of adding more report fields or benchmark-ingestion behavior.

### Risks

- Risk: broad `just check` still exposes legacy Ruff debt outside this slice.
- Mitigation: run focused validation plus `just imports`, and avoid legacy Ruff
  cleanup unless explicitly requested.

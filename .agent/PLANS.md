# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: refresh the all-five 32^3 benchmark summary after the verification gate.

### Scope

- In scope:
  - Rerun all five planned 32^3 sidecar benchmark cases after the
    Schur-accepted verification gate.
  - Render the comparison markdown from the refreshed `benchmark_result.json`
    files.
  - Update the tracked benchmark summary and implementation log with current
    timing/recovery results.
- Out of scope:
  - Adding or changing artifact/report/observability fields.
  - New benchmark ingestion behavior.
  - Solver tuning beyond command-line flags already supported.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` run artifacts with public
  `tomojax.datasets` sidecar generation/loading and `tomojax.bench` comparison.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Rerun all five 32^3 sidecar benchmark cases.
- [x] Render refreshed comparison markdown.
- [x] Update tracked benchmark summary and implementation log.
- [x] Run `just imports`.
- [x] Commit refreshed benchmark summary.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run python` generated five 32^3 sidecar datasets
  under `.artifacts/phase8_multi_case_32_after_accept_gate/datasets`.
- `JAX_PLATFORM_NAME=cpu uv run tomojax-align-auto-smoke ...` completed for all
  five existing sidecar directories.
- `uv run tomojax-synthetic-benchmark-compare ... --out .artifacts/phase8_multi_case_32_after_accept_gate/benchmark_comparison.md`
  passed.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Four rejected-Schur cases now report no verified geometry timing; the thermal
  drift case still reports a verified geometry time and supported-DOF
  improvement.

### Risks

- Risk: generated run artifacts remain ignored under `.artifacts/`.
- Mitigation: commit the concise markdown summary and implementation-log
  evidence.

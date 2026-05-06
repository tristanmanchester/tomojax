# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: replace synthetic benchmark timing placeholders with measured smoke-run
  timing fields.

### Scope

- In scope:
  - Measure total smoke-run wall time and time to first verified geometry.
  - Thread timing through verification, `benchmark_result.json`, and
    `benchmark_report.md`.
  - Add focused CLI coverage for non-null positive timing fields.
- Out of scope:
  - Stripe/ring bias fields.
  - Larger 128^3 benchmark runtime.
  - Detector-shift volume gauge correction.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add measured smoke timing to verification.
- [x] Thread timing into benchmark result/report.
- [x] Add focused timing assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the benchmark timing slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed: 5 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py -q`
  passed: 17 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Timing covers the smoke solver path up to artifact emission; full
  end-to-end CLI timing remains a later benchmark harness concern.

### Risks

- Risk: wall time varies across machines and test runs.
- Mitigation: assert only finite positive timing, not exact values.

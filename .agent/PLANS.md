# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: add report-only benchmark manifest criteria evaluation to smoke-run
  benchmark artifacts.

### Scope

- In scope:
  - Evaluate manifest criteria when the smoke run exposes a matching metric.
  - Mark unsupported criteria as `not_evaluated`.
  - Thread evaluation into `benchmark_result.json` and `benchmark_report.md`.
  - Add focused CLI assertions for evaluated and not-evaluated criteria.
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

- [x] Add report-only criteria evaluation payload.
- [x] Render criteria evaluation in markdown report.
- [x] Add focused criteria-evaluation assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the benchmark criteria-evaluation slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 7 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Criteria evaluation is report-only and must not change solver pass/fail
  behavior in this slice.

### Risks

- Risk: unsupported benchmark criteria can look silently ignored.
- Mitigation: emit an explicit `not_evaluated` status with a reason.

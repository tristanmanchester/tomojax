# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: add aggregate benchmark manifest criteria evaluation status to
  smoke-run benchmark artifacts.

### Scope

- In scope:
  - Summarise criteria evaluation counts by status.
  - Add an aggregate report-only status.
  - Render the aggregate status in `benchmark_report.md`.
  - Add focused CLI assertions for the summary.
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

- [x] Add criteria evaluation summary payload.
- [x] Render criteria summary in markdown report.
- [x] Add focused criteria-summary assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the benchmark criteria-summary slice.

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

- Criteria summary is report-only and must not change solver pass/fail behavior
  in this slice.

### Risks

- Risk: aggregate status can be mistaken for the solver verification status.
- Mitigation: label it as `benchmark_manifest_evaluation_summary`.

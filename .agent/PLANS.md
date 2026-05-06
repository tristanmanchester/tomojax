# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 verification semantics
- Goal: gate reported time-to-verified geometry on final synthetic geometry recovery.

### Scope

- In scope:
  - Keep `time_to_verified_geometry_seconds` null unless final geometry
    recovery passes.
  - Cover the stopped-reconstruction sidecar recovery-gap case.
  - Record the changed timing semantics and focused validation.
- Out of scope:
  - Adding or changing artifact/report/observability fields.
  - New benchmark ingestion behavior.
  - Solver tuning beyond command-line flags already supported.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` verification payloads.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Gate time-to-verified geometry on final geometry recovery.
- [x] Add focused stopped-reconstruction sidecar assertion.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the timing semantics slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 11 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- A transient accepted level is not enough to publish verified geometry timing
  when the final synthetic recovery gate fails.

### Risks

- Risk: this changes benchmark timing summaries.
- Mitigation: cover the recovery-gap sidecar and refresh benchmark summaries in
  a follow-up if needed.

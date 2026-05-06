# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 verification semantics
- Goal: make supported-DOF improvement tolerate already-good DOFs.

### Scope

- In scope:
  - Treat supported DOFs as acceptable when they started within tolerance and
    remain within tolerance.
  - Preserve strict improvement for supported DOFs that start outside tolerance.
  - Cover the geometry recovery payload behavior with focused smoke assertions.
- Out of scope:
  - Adding or changing artifact/report/observability fields.
  - New benchmark ingestion behavior.
  - Changing the default stopped-reconstruction solver path.
  - Solver tuning.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` verification payloads.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Update supported-DOF improvement semantics.
- [x] Add focused geometry recovery assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the verification semantics slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 12 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- A DOF that starts within tolerance should not force
  `supported_dofs_improved=false` merely because it cannot strictly improve.

### Risks

- Risk: this could hide regressions inside tolerance.
- Mitigation: keep per-DOF `*_improved` and `*_passed` fields unchanged; only
  the aggregate supported-DOF boolean uses acceptable-or-improved semantics.

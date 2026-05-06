# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: add report-only weak DOF decisions from Schur observability evidence.

### Scope

- In scope:
  - Add conservative decision entries for `det_v_px` and `theta_scale`.
  - Base decisions on available Schur curvature and accepted-step evidence.
  - Record thresholds and reasons in `observability_report.json`.
  - Update focused artifact tests.
- Out of scope:
  - Auto-activating `det_v_px` or `theta_scale`.
  - Correlation and validation-improvement gates beyond available smoke data.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add weak DOF decision helper.
- [x] Emit decisions and thresholds in observability artifacts.
- [x] Add focused smoke artifact assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the weak-DOF decision slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 7 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decisions are advisory/report-only until setup/pose parameter activation has
  a full validation-improvement gate.

### Risks

- Risk: smoke data only supports curvature and accepted-step evidence, not the
  full correlation and validation-improvement policy.
- Mitigation: mark missing evidence explicitly and keep decisions conservative.

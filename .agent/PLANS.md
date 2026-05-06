# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: replace the smoke observability placeholder with Schur-backed weak DOF evidence.

### Scope

- In scope:
  - Feed the last Schur result into `observability_report.json`.
  - Record Schur condition/eigenvalue evidence in the report.
  - Emit explicit `det_v_px` and `theta_scale` weak/frozen statuses.
  - Update focused artifact tests.
- Out of scope:
  - Auto-activating `det_v_px` or `theta_scale`.
  - Full correlation-based weak DOF policy.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Thread Schur result into observability report writer.
- [x] Replace placeholder weak-mode payload with Schur-backed evidence.
- [x] Add focused smoke artifact assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the observability slice.

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

- Treat this as evidence reporting only. Activation/deactivation decisions for
  weak DOFs remain a later Phase 8 slice.

### Risks

- Risk: curvature evidence is coarse because supported Schur setup parameters
  are only theta offset, det_u, and active det_v.
- Mitigation: report status and reasons explicitly rather than pretending full
  weak-DOF policy is complete.

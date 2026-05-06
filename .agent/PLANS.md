# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 nuisance models and weak DOF handling
- Goal: make the frozen `theta_scale` weak-DOF report evidence explicit.

### Scope

- In scope:
  - Add explicit missing-evidence labels for frozen `theta_scale` curvature,
    correlation, accepted-step, and validation-improvement evidence.
  - Preserve the report-only weak-DOF policy and public alternating smoke API.
  - Cover the artifact shape in the deterministic alternating smoke assertions.
- Out of scope:
  - New nuisance solver blocks.
  - Automatic weak-DOF activation changes.
  - Enabling `theta_scale` optimisation in the projector or solver.
  - Further report scaffold polishing beyond the missing-evidence payload.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` observability/report payloads.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add explicit missing-evidence labels for frozen `theta_scale`.
- [x] Add focused alternating smoke assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the theta-scale evidence slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed: 2 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep `theta_scale` frozen in report-only mode until the reference projector
  supports an identifiable scale parameter.

### Risks

- Risk: the missing-evidence vocabulary can drift from active DOF evidence.
- Mitigation: assert frozen `theta_scale` evidence alongside active `det_v_px`
  evidence in the smoke artifact test.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 active-DOF observability reporting slice
- Goal: make `observability_report.json` reflect the Schur setup and pose DOFs
  that actually ran, so supported geometry DOFs are not reported as
  `weak_not_evaluated` after they are wired into the active solver block.

### Scope

- In scope:
  - Derive setup DOF `active`/`status`/`observable` from
    `JointSchurLMResult.active_setup_parameters`.
  - Derive pose DOF `active`/`status`/`observable` from
    `JointSchurLMResult.active_pose_dofs`, preserving gauge-canonicalised
    status for active `phi_residual_rad`, `dx_px`, and `dz_px`.
  - Add focused alternating-smoke assertions for active theta/setup and active
    pose observability payloads.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Changing solver maths, benchmark tolerances, or adding new report fields.
  - Rerunning the 128^3 scale gate.
- Deep module owner: `tomojax.align` for benchmark artifact/report evaluation.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Update observability DOF status generation from Schur active blocks.
- [x] Add focused alternating-smoke tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_observability.py
  -q` passed: 2 tests in 0.81 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 1 test in 112.65 seconds.
- `uv run ruff check src/tomojax/align/_alternating_verification.py
  tests/test_alternating_observability.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py
  tests/test_alternating_observability.py tests/test_alternating_solver_smoke.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- Supported DOFs that are in the active Schur block must not be reported as
  `weak_not_evaluated` in observability artifacts.

### Completed Previous Slices

- [x] Detector roll supported and committed: `2be6a99`.
- [x] Axis tilt supported and committed with GPU diagnostic pause:
  `ac347d2`.
- [x] Alpha/beta pose supported and committed: `aea525d`.
- [x] Supported geometry update DOFs exposed in `align-auto`: `19dd503`.
- [x] Theta-scale opt-in setup updates committed: `be3d059`.
- [x] Parallel laminography acquisition metadata committed: `7aa086c`.
- [x] det_v observability gating evidence committed: `7c1e0fe`.
- [x] Synthetic unsupported-term classification committed: `28e336f`.
- [x] Benchmark criterion aliases committed: `fe83427`.
- [x] Laminography solver residuals committed: `7002d42`.
- [x] Recovered det_v policy criterion committed: `f6fe3c4`.
- [x] Backend policy criterion evaluation committed: `b040829`.
- [x] Calibrated-grid backend provenance committed: `a0b69db`.
- [x] Missing-policy criterion reasons committed: `9034b91`.
- [x] 128^3 supported-only GPU scale gate committed: `d2fbd5a`.

### Risks

- Risk: active status can be mistaken for an observability decision.
- Mitigation: active Schur DOFs get `status=evaluated` only when diagnostics
  exist; weak-DOF policy decisions remain separate and report-only.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 missing-policy criterion reason slice
- Goal: replace generic unsupported-DOF reasons for object-motion, bad-view,
  jump-exclusion, and baseline-comparison benchmark criteria with explicit
  missing-policy evidence reasons.

### Scope

- In scope:
  - Add explicit criterion evaluation branches for `core_solver`,
    `bad_views_flagged`, `pose_dx_dz_rmse_px_lt_except_jumps`,
    `beats_current_default_nmse`, and `object_motion_enabled_tx_rmse_px_lt`.
  - Keep these criteria `not_evaluated` until their required evidence payloads
    exist.
  - Add focused criterion-evaluator tests.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Implementing object-motion solvers, bad-view detection, jump exclusion
    metrics, current-default comparisons, or benchmark reruns.
- Deep module owner: `tomojax.align` for benchmark artifact/report evaluation.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add explicit missing-evidence criterion branches.
- [x] Add focused criterion-evaluator tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_benchmark_criteria.py -q` passed: 10 tests in
  0.65 seconds.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_benchmark_criteria.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_benchmark_criteria.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- Unsupported criteria may remain only with exact missing evidence reasons, not
  generic unsupported placeholders.

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

### Risks

- Risk: explicit missing-evidence branches can look like implementation.
- Mitigation: keep status `not_evaluated` and name the absent payload required
  to make each criterion real.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 benchmark manifest criterion alias slice
- Goal: evaluate documented benchmark manifest geometry criteria aliases against
  existing recovery metrics so supported roll/axis criteria no longer fall into
  `unsupported_dof_not_evaluated` purely because of naming drift.

### Scope

- In scope:
  - Evaluate `detector_roll_error_deg_lt` as the existing
    `detector_roll_error_rad` metric.
  - Evaluate `axis_roll_error_deg_lt` as the max of existing axis and detector
    roll error metrics.
  - Keep string policy criteria report-only/not-evaluated until their policy
    payloads exist.
  - Add focused criterion-evaluator tests.
  - Update docs/logs and commit the slice.
- Out of scope:
  - New benchmark result fields, policy evaluation payloads, tolerance changes,
    or benchmark reruns.
- Deep module owner: `tomojax.align` for benchmark artifact/report evaluation.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add detector-roll criterion alias evaluation.
- [x] Add axis+roll combined criterion evaluation.
- [x] Add focused criterion-evaluator tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_benchmark_criteria.py -q` passed: 3 tests in
  0.67 seconds.
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
- Criteria that map to existing recovery metrics should be evaluated under all
  documented aliases. Policy criteria remain not evaluated until a real policy
  payload is available.

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

### Risks

- Risk: combined criteria can hide which component failed.
- Mitigation: keep the value as the max component error and leave individual
  axis/roll criteria evaluable where the manifest names them separately.

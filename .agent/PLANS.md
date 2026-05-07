# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 det-v policy criterion evidence slice
- Goal: make the existing `det_v_policy` benchmark criterion use the
  weak-DOF policy decision that is already emitted in observability artifacts,
  instead of remaining `not_evaluated` whenever det_v is not recovered.

### Scope

- In scope:
  - Pass the report-only weak-DOF policy payload into benchmark manifest
    criterion evaluation.
  - Treat `det_v_policy=recovered_or_reported_unobservable` as passed when det_v
    is either recovered or the policy decision says to keep it frozen.
  - Add focused criterion evaluator tests.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Changing solver maths, benchmark tolerances, or report payload shape.
  - Promoting weak-DOF policy from report-only to enforcement.
- Deep module owner: `tomojax.align` for benchmark artifact/report evaluation.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Wire weak-DOF policy into benchmark criterion evaluation.
- [x] Add focused det_v policy tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_benchmark_criteria.py -q` passed: 11 tests in
  0.63 seconds.
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
- Existing weak-DOF policy evidence should make policy criteria real when it is
  sufficient, without adding duplicate benchmark fields.

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
- [x] Active Schur DOFs in observability committed: `7ab5013`.
- [x] Smoke expectation cleanup committed: `44dda7e`.
- [x] Nuisance-corrected failure gate committed: `f374d58`.

### Risks

- Risk: a report-only policy pass can look like det_v numerical recovery.
- Mitigation: keep criterion reasons explicit: numerical recovery and
  reported-unobservable/frozen policy are distinct pass reasons.

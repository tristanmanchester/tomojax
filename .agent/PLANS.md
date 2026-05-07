# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 det_v policy criterion evaluation slice
- Goal: make the documented `det_v_policy: recovered_or_reported_unobservable`
  benchmark criterion evaluate against existing det_v recovery evidence when
  recovery is available, while preserving not-evaluated status when the
  unobservability report payload is not yet present.

### Scope

- In scope:
  - Evaluate `det_v_policy` as passed when `det_v_realized_rmse_px_passed` is
    true.
  - Leave `det_v_policy` not evaluated with a specific reason when det_v is not
    recovered and no unobservability policy payload is available in
    `benchmark_result`.
  - Add focused criterion-evaluator tests.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Adding new benchmark result fields, wiring full observability reports into
    benchmark_result, backend policy evaluation, or benchmark reruns.
- Deep module owner: `tomojax.align` for benchmark artifact/report evaluation.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Evaluate recovered det_v policy criteria.
- [x] Preserve not-evaluated status when unobservability evidence is missing.
- [x] Add focused criterion-evaluator tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_benchmark_criteria.py -q` passed: 5 tests in
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
- Policy criteria should only pass from concrete evidence. Missing policy
  payloads remain `not_evaluated` with a specific reason.

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

### Risks

- Risk: policy criteria could become false positives if treated as ordinary
  numeric metrics.
- Mitigation: keep policy evaluation explicit and pass only when the recovered
  branch is directly evidenced.

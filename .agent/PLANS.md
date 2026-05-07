# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 smoke expectation cleanup slice
- Goal: align slow alternating-smoke assertions with current synthetic sidecar
  contracts so broad smoke validation no longer requires unsupported nuisance,
  roll, or axis recovery to pass.

### Scope

- In scope:
  - Relax sidecar-ingestion smoke assertions from whole-geometry pass to
    individual supported-DOF evidence where the synthetic scenario contains
    currently unsupported nuisance/roll/axis terms.
  - Keep stopped-reconstruction recovery-gap assertions focused on the recovery
    gap and stopped-volume gauge payload shape, not a fixed nearest-geometry
    label.
  - Record the focused failing/passing smoke commands.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Changing solver maths, benchmark tolerances, or report payload shape.
  - Reclassifying unsupported synthetic scenario terms as supported.
- Deep module owner: `tomojax.align` for benchmark artifact/report evaluation.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Update stale slow-smoke assertions.
- [x] Rerun the three previously failing smoke tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_ingests_generated_synthetic_sidecars
  tests/test_alternating_solver_smoke.py::test_alternating_solver_stopped_reconstruction_sidecar_reports_recovery_gap
  tests/test_alternating_solver_smoke.py::test_supported_dof_summary_reports_individual_dof_evidence
  -q` passed: 3 tests in 335.52 seconds.
- `uv run ruff check tests/test_alternating_solver_smoke.py` passed.
- `uv run basedpyright tests/test_alternating_solver_smoke.py` passed with
  0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- Smoke tests for unsupported synthetic terms should assert explicit individual
  evidence, not accidental whole-run success.

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

### Risks

- Risk: relaxing assertions can hide real regressions.
- Mitigation: keep assertions on deterministic sidecar ingestion, finite traces,
  individual supported-DOF metrics, and stopped-volume gauge payload shape.

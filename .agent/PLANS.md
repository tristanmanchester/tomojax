# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 2/7 laminography solver residual vertical slice
- Goal: thread acquisition nominal-axis metadata into setup-only LM, pose-only
  LM, and joint Schur LM residual/loss paths so laminography sidecars use the
  supported core laminography geometry during actual geometry updates, not only
  during direct state projection.

### Scope

- In scope:
  - Expose a typed forward public helper for nominal axis derivation from
    `GeometryState`.
  - Pass laminography nominal axis into setup-only LM, pose-only LM, and joint
    Schur LM array-projector residuals.
  - Add focused zero-residual tests for laminography acquisition in each solver
    path.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Solving laminography tilt as an active setup parameter or rerunning
    benchmarks.
- Deep module owners: `tomojax.forward` for public acquisition-to-axis
  semantics and `tomojax.align` for solver residual paths.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Expose nominal axis derivation through `tomojax.forward`.
- [x] Thread acquisition nominal axis into LM/Schur residual projectors.
- [x] Add focused solver zero-residual tests for laminography acquisition.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_setup_lm.py::test_setup_only_lm_residual_preserves_laminography_acquisition
  tests/test_pose_lm.py::test_pose_only_lm_residual_preserves_laminography_acquisition
  tests/test_joint_schur_lm.py::test_joint_schur_lm_residual_preserves_laminography_acquisition
  -q` passed: 3 tests in 5.31 seconds.
- `uv run ruff check ...` passed for touched forward/align source and solver
  tests.
- `uv run basedpyright ...` passed with 0 errors, 0 warnings, and 0 notes for
  touched forward/align source and solver tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- The only operational projector path remains `core_trilinear_ray`; solver
  residuals must use the same acquisition nominal axis as direct
  `GeometryState` projection.

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

### Risks

- Risk: direct projection tests can pass while solver residuals still use the
  default parallel nominal axis.
- Mitigation: add zero-residual solver tests using laminography acquisition
  side-by-side with direct `project_parallel_reference` observations.

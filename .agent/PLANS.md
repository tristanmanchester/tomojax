# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 2 per-view alpha/beta pose geometry vertical slice
- Goal: promote `alpha_rad` and `beta_rad` from unsupported/frozen placeholders
  to real per-view pose DOFs for the `core_trilinear_ray` v2 path, including
  adapter semantics, pose/Schur packing, sidecar classification, artifacts,
  docs, and focused tests.

### Scope

- In scope:
  - Apply `GeometryState.pose.alpha_rad` and `beta_rad` through the v2-to-core
    pose stack while preserving the single `core_trilinear_ray` projector
    family.
  - Extend pose-only LM and joint Schur LM active pose packing/unpacking to
    include alpha/beta as opt-in pose DOFs.
  - Stop stripping/classifying alpha and beta as unsupported in generated
    synthetic sidecars.
  - Add focused CPU tests for alpha/beta projection semantics and pose/Schur
    recovery on deterministic asymmetric phantoms.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Parallel laminography, theta-scale activation, det_v gating, object drift,
    projector selectors, threshold relaxation, legacy Ruff cleanup, or full
    benchmark reruns.
- Deep module owners: `tomojax.forward` for core adapter semantics,
  `tomojax.align` for pose/Schur packing and recovery reporting, and
  `tomojax.datasets` for sidecar unsupported-DOF classification.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Wire `alpha_rad` and `beta_rad` into `core_projection_geometry_from_state`
  and `project_parallel_reference_arrays`.
- [x] Extend pose-only and joint Schur active pose DOFs with alpha/beta.
- [x] Remove alpha/beta from unsupported sidecar projection/classification.
- [x] Add focused adapter and LM tests for alpha/beta pose rotations.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the alpha/beta slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_forward_reference.py
  tests/test_pose_lm.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_recovers_realized_supported_geometry
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_alpha_beta_pose_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_axis_tilt_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_detector_roll_setup_update
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_generates_named_synthetic_dataset
  tests/test_synthetic_datasets.py::test_generate_synthetic_dataset_writes_deterministic_smoke_artifacts
  tests/test_synthetic_datasets.py::test_generate_supported_only_setup_global_dataset_removes_unsupported_truth
  -q` passed: 27 tests in 278.34 seconds.
- `uv run ruff check ...` passed for touched source and tests.
- `uv run basedpyright ...` passed with 0 errors and 0 warnings for touched
  source and tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- Axis tilt convention reuses `tomojax.calibration.axis_geometry`: derive a
  lab-frame axis unit from x/y axis rotations, build world-from-object `T_all`
  with `axis_pose_stack`, then apply supported detector centre/roll semantics
  independently.
- Alpha/beta convention for this slice follows the sidecar geometry wrapper:
  build the nominal axis/theta world-from-object pose, then compose per-view
  alpha/beta residual rotations in object coordinates. The existing `dx_px` and
  `dz_px` detector-shift sign convention remains unchanged.

### Completed Previous Slices

- [x] Detector roll supported and committed: `2be6a99`.
- [x] Axis tilt supported and committed with GPU diagnostic pause:
  `ac347d2`.

### Risks

- Risk: alpha/beta are weak and gauge-coupled with axis tilt on small or
  symmetric phantoms.
- Mitigation: keep alpha/beta opt-in in solver configs, use asymmetric focused
  tests, and report any weak-mode observability issues explicitly.

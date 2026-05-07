# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 2/5 axis-tilt geometry vertical slice
- Goal: promote `axis_rot_x_rad` and `axis_rot_y_rad` from unsupported to real
  core-trilinear v2 setup DOFs for parallel tomography, including adapter
  semantics, setup/Schur packing, sidecar classification, artifacts, docs, and
  focused tests.

### Scope

- In scope:
  - Apply `GeometryState.setup.axis_rot_x_rad` and `axis_rot_y_rad` through the
    v2-to-core pose stack while preserving the single `core_trilinear_ray`
    projector family.
  - Add axis-tilt setup packing/unpacking for setup-only and joint Schur LM.
  - Stop classifying axis rotations as unsupported in generated benchmark
    sidecars.
  - Add focused CPU tests for axis-tilt projection semantics and Schur/setup
    recovery on a deterministic asymmetric phantom.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Laminography, alpha/beta pose, object drift, projector selectors,
    threshold relaxation, legacy Ruff cleanup, or full five-case reruns.
- Deep module owners: `tomojax.forward` for detector-grid adapter semantics,
  `tomojax.align` for setup/Schur packing, and `tomojax.datasets` for sidecar
  unsupported-DOF classification.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Wire `axis_rot_x_rad` and `axis_rot_y_rad` into
  `core_projection_geometry_from_state` and `project_parallel_reference_arrays`.
- [x] Extend setup-only and joint Schur active setup parameters with
  `axis_rot_x_rad` and `axis_rot_y_rad`.
- [x] Remove axis rotations from unsupported sidecar projection/classification.
- [x] Add focused adapter and LM tests for axis rotations.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the axis-tilt slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_forward_reference.py
  tests/test_setup_lm.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_recovers_realized_supported_geometry
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_freeze_pose_dofs_for_setup_oracle
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_det_u_only_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_detector_roll_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_axis_tilt_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_writes_normal_eq_summary_artifact
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_generates_named_synthetic_dataset
  tests/test_synthetic_datasets.py::test_generate_synthetic_dataset_writes_deterministic_smoke_artifacts
  tests/test_synthetic_datasets.py::test_generate_supported_only_setup_global_dataset_removes_unsupported_truth
  -q` passed: 27 tests in 261.43 seconds.
- `uv run ruff check ...` passed for touched source and tests.
- `uv run basedpyright ...` passed with 0 errors and 0 warnings for touched
  source and tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- New decision: the v2 rotate-and-sum projector is no longer an operational
  projector path. The only supported v2 operator family is the existing core
  trilinear ray projector/backprojector (`core_trilinear_ray`).
- Do not add a long-term selector between rotate-and-sum and core trilinear ray.
  Any old rotate-sum behavior may remain only as deleted history or a narrowly
  private test fixture if unavoidable.
- Axis tilt convention will reuse `tomojax.calibration.axis_geometry`
  semantics: derive a lab-frame axis unit from x/y axis rotations, build
  world-from-object `T_all` with `axis_pose_stack`, then apply supported
  detector centre/roll semantics independently.

### Completed Previous Slice

- [x] Wire `detector_roll_rad` into `core_projection_geometry_from_state` and
  `project_parallel_reference_arrays`.
- [x] Extend setup-only and joint Schur active setup parameters with
  `detector_roll_rad`.
- [x] Remove detector roll from unsupported sidecar projection/classification.
- [x] Add focused adapter and LM tests for detector roll.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the detector-roll slice:
  `2be6a99`.

### Risks

- Risk: current recovery metrics were calibrated against the toy projector and
  may regress once the physical core ray operator is used.
- Mitigation: rebaseline fixed-truth first; if fixed-truth fails, treat it as an
  adapter/scaling blocker before interpreting stopped reconstruction quality.

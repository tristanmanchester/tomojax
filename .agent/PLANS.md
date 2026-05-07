# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 theta-scale active-with-prior vertical slice
- Goal: make `theta_scale` a supported opt-in setup parameter for setup-only
  and joint Schur LM, expose it through `align-auto`, and report/evaluate its
  recovery while preserving the default frozen/observability-gated policy.

### Scope

- In scope:
  - Add `theta_scale` to setup-only LM packing/unpacking and frozen-parameter
    reporting.
  - Add `theta_scale` to joint Schur setup packing/unpacking, diagnostics, and
    frozen-parameter reporting.
  - Allow `theta_scale` in alternating geometry-update validation and
    `align-auto` active setup parameter parsing/help.
  - Add `theta_scale_error` recovery metrics and manifest criterion evaluation.
  - Add focused CPU tests for opt-in setup-only and joint Schur theta-scale
    recovery on identifiable nonzero theta spans.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Automatic weak-DOF activation, changing default schedules, det_v policy
    changes, parallel laminography, benchmark reruns, or threshold relaxation.
- Deep module owners: `tomojax.align` for setup/Schur packing and reporting,
  `tomojax.cli` for CLI parsing, and `tomojax.forward` only indirectly through
  the existing `theta_total_rad`/array theta path.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add `theta_scale` to setup-only LM active setup parameters.
- [x] Add `theta_scale` to joint Schur active setup parameters.
- [x] Expose `theta_scale` through alternating validation and `align-auto`.
- [x] Add theta-scale recovery/manifest metrics.
- [x] Add focused setup-only, Schur, and CLI tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_setup_lm.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_recovers_realized_supported_geometry
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_det_u_only_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_detector_roll_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_axis_tilt_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_theta_scale_setup_update
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  tests/test_align_auto_cli.py::test_align_auto_parses_supported_geometry_update_dofs
  tests/test_align_auto_cli.py::test_align_auto_rejects_unknown_geometry_update_dofs
  -q` passed: 14 tests in 113.33 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  tests/test_setup_lm.py::test_setup_only_lm_recovers_theta_scale_when_explicitly_active
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_theta_scale_setup_update
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  tests/test_align_auto_cli.py::test_align_auto_parses_supported_geometry_update_dofs
  tests/test_align_auto_cli.py::test_align_auto_rejects_unknown_geometry_update_dofs
  -q` passed: 6 tests in 125.98 seconds.
- `uv run ruff check ...` passed for touched Python source and tests.
- `uv run basedpyright ...` passed with 0 errors and 0 warnings for touched
  source and tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- `theta_scale` remains frozen by default. This slice only permits explicit
  activation with priors/trust radii so observability policy can decide when it
  is safe.

### Completed Previous Slices

- [x] Detector roll supported and committed: `2be6a99`.
- [x] Axis tilt supported and committed with GPU diagnostic pause:
  `ac347d2`.
- [x] Alpha/beta pose supported and committed: `aea525d`.
- [x] Supported geometry update DOFs exposed in `align-auto`: `19dd503`.

### Risks

- Risk: theta scale is weak or nearly collinear with theta offset on tiny or
  narrow-span smoke datasets.
- Mitigation: keep it opt-in, test with a nonzero theta span and setup-only
  pose-frozen geometry, and keep default observability report status frozen
  unless it is explicitly active in Schur diagnostics.

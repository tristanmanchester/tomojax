# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7/CLI geometry-update activation follow-up
- Goal: expose the newly supported setup and pose DOFs through `align-auto`
  parsing/help and alternating private validation so benchmark diagnostics can
  activate detector roll, axis tilt, and alpha/beta without bypassing the CLI.

### Scope

- In scope:
  - Update `align-auto` active pose DOF parsing/help for
    `alpha_rad,beta_rad,phi_residual_rad,dx_px,dz_px`.
  - Update `align-auto` active setup parsing/help for detector roll and axis
    x/y tilt.
  - Update alternating geometry-update validation to accept the same active DOF
    names that joint Schur supports.
  - Add focused CLI/unit tests for accepted and rejected active DOF lists.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Changing default schedules, activating weak DOFs automatically, parallel
    laminography, theta-scale activation, det_v gating, or benchmark reruns.
- Deep module owners: `tomojax.cli` for CLI parsing and `tomojax.align` for
  private alternating validation.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Update `align-auto` active pose DOF parser/help.
- [x] Update `align-auto` active setup parser/help.
- [x] Update alternating geometry-update private validation.
- [x] Add focused parser/CLI tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the CLI activation slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  tests/test_align_auto_cli.py::test_align_auto_parses_supported_geometry_update_dofs
  tests/test_align_auto_cli.py::test_align_auto_rejects_unknown_geometry_update_dofs
  -q` passed: 3 tests in 0.67 seconds.
- `uv run ruff check src/tomojax/cli/align_auto.py
  src/tomojax/align/_alternating_geometry_update.py
  tests/test_align_auto_cli.py` passed.
- `uv run basedpyright src/tomojax/cli/align_auto.py
  src/tomojax/align/_alternating_geometry_update.py
  tests/test_align_auto_cli.py` passed with 0 errors, 0 warnings, and 0 notes.
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
- [x] Alpha/beta pose supported and committed: `aea525d`.

### Risks

- Risk: alpha/beta are weak and gauge-coupled with axis tilt on small or
  symmetric phantoms.
- Mitigation: keep alpha/beta opt-in in solver configs, use asymmetric focused
  tests, and report any weak-mode observability issues explicitly.

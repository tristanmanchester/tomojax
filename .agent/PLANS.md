# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 6 — Joint setup+pose Schur LM
- Goal: add the first joint setup+pose Schur LM reference slice for supported
  setup and pose DOFs.

### Scope

- In scope:
  - Pack supported setup DOFs plus per-view pose DOFs into one joint problem.
  - Build finite-difference residual Jacobians and damped normal equations.
  - Solve setup step by Schur complement over per-view pose blocks.
  - Add diagnostics for Schur condition and update norms.
  - Add deterministic recovery and Schur-vs-dense tests.
- Out of scope:
  - Alpha/beta pose effects.
  - Detector roll, axis rotations, theta scale.
  - Priors, trust radii, and full acceptance policy.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Add joint setup+pose Schur LM reference implementation.
- [x] Export the public joint solver API.
- [x] Add deterministic recovery and Schur-vs-dense tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [x] Commit the joint Schur LM slice if validations pass.

### Follow-Up Slice

- [x] Add `normal_eq_summary.json` artifact writer for joint Schur diagnostics.
- [x] Add artifact readback test.
- [x] Re-run validation commands for the artifact contract.
- [x] Add global/Schur eigenvalues and pose-block condition diagnostics.
- [x] Re-run validation commands for enriched diagnostics.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 11 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 147 tests.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: start with supported differentiable DOFs only:
  `theta_offset_rad`, `det_u_px`, active `det_v_px`,
  `phi_residual_rad`, `dx_px`, and `dz_px`.
- Deviation: this is a reference Schur slice, not the final production
  trust-region solver.

### Risks

- Risk: setup and pose gauge pairs are non-identifiable without
  canonicalisation.
- Mitigation: tests assert realised geometry after canonicalisation, not raw
  uncanonicalized parameter equality for gauge-coupled values.

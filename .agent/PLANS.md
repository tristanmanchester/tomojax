# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 4 — Pose-only 5x5 LM
- Goal: extend the current pose-only LM from detector shifts to include
  differentiable per-view `phi_residual_rad`.

### Scope

- In scope:
  - Add `phi_residual_rad` to the pose-only LM packed parameter vector.
  - Keep `alpha_rad` and `beta_rad` frozen.
  - Add deterministic phi recovery and gauge-canonicalisation tests.
  - Update align README and implementation log.
- Out of scope:
  - Alpha/beta 3D tilt pose effects.
  - Trust radii, damping adaptation, and Schur coupling.
  - Held-out residual validation.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Extend pose-only LM to optimise phi.
- [x] Add deterministic phi recovery tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the pose-phi LM slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_pose_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_pose_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_pose_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 20 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 143 tests.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: optimise `phi_residual_rad`, `dx_px`, and `dz_px` with the existing
  finite-difference LM normal equation.
- Deviation: this is still not the full 5-DOF pose solver because alpha/beta
  require projector support for out-of-plane pose effects.

### Risks

- Risk: theta and detector shifts can partially trade off on tiny asymmetric
  volumes.
- Mitigation: use deterministic small perturbations and assert active/frozen
  DOF reporting plus gauge canonicalisation.

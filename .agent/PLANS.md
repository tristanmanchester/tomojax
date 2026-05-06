# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 2 — JAX reference forward model and residual loss
- Goal: make the minimal parallel reference projector differentiable with
  respect to view angle, unlocking later phi/theta optimisation work.

### Scope

- In scope:
  - Replace quadrant `rot90` projection with bilinear in-plane rotation.
  - Preserve current detector-shift projection behavior.
  - Add deterministic tests for theta sensitivity and autodiff gradient.
  - Update forward README and implementation log.
- Out of scope:
  - Laminography geometry.
  - Detector roll, axis rotations, and full 5-DOF pose effects.
  - Pallas/GPU fast paths.
- Deep module owner: `tomojax.forward`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Implement differentiable angle projection.
- [x] Add theta sensitivity and gradient tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the differentiable-theta projector slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/forward/_projector.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward/_projector.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/forward/_projector.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py -q`
  passed: 20 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 141 tests.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: use bilinear sampling in the x-y plane with zero outside-volume
  boundaries for the reference projector.
- Deviation: this remains a minimal parallel reference model, not the full
  physical tomography/laminography projector.

### Risks

- Risk: changing boundary behavior from quadrant rotation can affect smoke
  numerics.
- Mitigation: keep existing detector shift tests and run the targeted v2
  regression bundle.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 5 precursor — pose-only geometry optimisation
- Goal: add the first pose-only LM/GN solver against a fixed volume for the
  detector-shift pose components supported by the current reference projector.

### Scope

- In scope:
  - Add a typed pose-only LM result/report.
  - Optimise per-view `dx_px` and `dz_px` using a damped Gauss-Newton/LM normal
    equation against fixed-volume projection residuals.
  - Use masked whitened projection residuals and pseudo-Huber IRLS weights.
  - Canonicalise gauges after the accepted solve.
  - Add deterministic synthetic recovery tests for `dx_px`/`dz_px`.
- Out of scope:
  - Optimising `alpha_rad`, `beta_rad`, and `phi_residual_rad`; the current
    projector does not yet provide physical differentiable sensitivity for
    those DOFs.
  - Setup-only and combined Schur setup+pose optimisation.
  - Full finite-difference Jacobian and Schur-vs-dense test suite.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Add pose-only LM/GN implementation.
- [x] Export the public solver/report API.
- [x] Add deterministic recovery tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the pose-only detector-shift solver slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_pose_lm.py
  src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_pose_lm.py
  tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py` passes.
- `uv run basedpyright src/tomojax/align/_pose_lm.py
  src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_pose_lm.py
  tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py` passes with
  0 errors and 0 warnings.
- `uv run pytest tests/test_pose_lm.py tests/test_vertical_smoke.py
  tests/test_v2_module_skeleton.py -q` passes with 7 tests.
- `uv run ruff format --check src/tomojax/align/_pose_lm.py
  src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_pose_lm.py
  tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py` passes.
- `just imports` passes.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py
  tests/test_align_checkpoint.py tests/test_axes_io.py
  tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py
  tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py
  tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py
  tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py
  tests/test_geometry_gauges.py tests/test_geometry_serialization.py
  tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py
  -q` passes with 128 tests.
- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup log.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: implement the first pose-only solver for `dx_px` and `dz_px` only,
  because those are the differentiable pose channels in the current reference
  projector.
- Deviation: this is not yet the full 5-DOF pose solver required by the final
  numerical plan.

### Risks

- Risk: this could be mistaken for complete pose-only optimisation.
- Mitigation: reports expose `active_dofs` and `frozen_dofs`, and docs/log mark
  alpha/beta/phi as pending physical projector support.

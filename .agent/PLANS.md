# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 5 precursor — setup-only geometry optimisation
- Goal: add the first setup-only LM/GN solver for detector-shift setup
  parameters supported by the current reference projector.

### Scope

- In scope:
  - Add a typed setup-only LM result/report.
  - Optimise setup `det_u_px` and active `det_v_px` using a damped
    Gauss-Newton/LM normal equation against fixed-volume projection residuals.
  - Use masked whitened residuals and pseudo-Huber IRLS weights.
  - Add deterministic synthetic recovery tests.
- Out of scope:
  - Optimising detector roll, axis rotations, theta offset, or theta scale.
  - Combined Schur setup+pose optimisation.
  - Full finite-difference/Schur-vs-dense test suite.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Add setup-only LM/GN implementation.
- [x] Export the public solver/report API.
- [x] Add deterministic recovery tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the setup-only detector-shift solver slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_lm_numerics.py src/tomojax/align/_setup_lm.py src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_lm_numerics.py src/tomojax/align/_setup_lm.py src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_lm_numerics.py src/tomojax/align/_setup_lm.py src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 6 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 130 tests.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: implement setup-only optimisation for `det_u_px` and active
  `det_v_px` only, because these are currently supported by the differentiable
  reference projector.
- Deviation: this is not yet the full setup solver required by the final
  numerical plan.

### Risks

- Risk: this could be mistaken for complete setup-only optimisation.
- Mitigation: reports expose active and frozen setup parameters.

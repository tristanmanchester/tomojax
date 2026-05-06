# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 5 — Setup-only LM
- Goal: extend setup-only LM to include global `theta_offset_rad` now that the
  reference projector is differentiable in theta.

### Scope

- In scope:
  - Add `theta_offset_rad` to the setup-only LM packed parameter vector.
  - Keep detector roll, axis rotations, and theta scale frozen.
  - Add deterministic theta-offset recovery tests.
  - Update align README and implementation log.
- Out of scope:
  - Detector roll and axis-direction setup gradients.
  - Setup observability diagnostics.
  - Schur joint setup+pose coupling.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Extend setup-only LM to optimise theta offset.
- [x] Add deterministic theta-offset recovery tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the setup-theta LM slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_setup_lm.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_setup_lm.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_setup_lm.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py -q`
  passed: 21 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 144 tests.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: optimise `theta_offset_rad`, `det_u_px`, and active `det_v_px` with
  the existing finite-difference LM normal equation.
- Deviation: detector roll, axis rotations, and theta scale remain frozen until
  the reference projector models those effects.

### Risks

- Risk: global theta and per-view phi remain a gauge pair.
- Mitigation: setup-only LM keeps pose fixed and uses the existing gauge
  canonicalisation report after solving.

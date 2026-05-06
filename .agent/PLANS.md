# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 6 — Joint setup+pose Schur LM
- Goal: add first trust-radius mechanics and diagnostics to the joint Schur
  reference solver.

### Scope

- In scope:
  - Add optional setup and pose update trust radii.
  - Clip accepted Schur steps by the configured radii.
  - Record trust scale, clipping status, and per-DOF update diagnostics.
  - Add deterministic tests and artifact readback coverage.
- Out of scope:
  - Adaptive trust-region radius update.
  - Prior terms and bounds.
  - Unsupported physical DOFs.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Add optional trust radii to joint Schur config.
- [x] Apply trust scaling to Schur and dense-equivalent steps.
- [x] Add trust and per-DOF update diagnostics to artifact payload.
- [x] Add deterministic tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the trust-radius slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 6 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 148 tests.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: use scalar setup/pose block radii for the first reference
  implementation, with per-DOF update diagnostics recorded separately.
- Deviation: this is not yet an adaptive trust-region policy.

### Risks

- Risk: overly small trust radii can slow convergence.
- Mitigation: radii default to `None`, preserving the existing full Schur step
  unless explicitly configured.

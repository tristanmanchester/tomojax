# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 6 — Joint setup+pose Schur LM
- Goal: add actual-vs-predicted reduction diagnostics to the joint Schur
  reference solver.

### Scope

- In scope:
  - Compute the damped quadratic model predicted reduction for the final Schur
    step after trust scaling.
  - Record actual candidate reduction and actual/predicted ratio.
  - Add deterministic tests and artifact readback coverage.
- Out of scope:
  - Adaptive trust-region radius update from the reduction ratio.
  - Prior terms and bounds.
  - Unsupported physical DOFs.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Compute predicted reduction from the Schur normal-equation model.
- [x] Record actual reduction and reduction ratio in diagnostics.
- [x] Add deterministic tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the reduction-diagnostics slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 149 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; current first failures are broad
  transitional legacy Ruff findings including `RUF002` in
  `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912` in
  `src/tomojax/align/_config.py`, and many remaining legacy lint issues.
  The formatter churn from this command was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: store `reduction_ratio` as `None` when the predicted reduction is
  numerically zero, which occurs naturally at convergence.
- Deviation: the ratio is diagnostic-only in this slice and does not yet drive
  trust-radius or damping policy.

### Risks

- Risk: the predicted reduction is from the current IRLS quadratic model, while
  actual reduction comes from the robust nonlinear objective.
- Mitigation: record both raw values and the ratio, and defer policy decisions
  to a later trust-region update slice.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  dedicated milestone rather than mixing repository-wide lint churn into Phase
  6 numerical solver slices.

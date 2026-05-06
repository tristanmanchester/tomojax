# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 3 — Reconstruction step: Huber-TV/FISTA preview
- Goal: add v2 reference reconstruction schedules for preview and final FISTA
  runs.

### Scope

- In scope:
  - Add typed v2 reference FISTA schedule entries.
  - Define preview levels 4 and 2 plus final level 1 schedules.
  - Include level-aware residual filter configs in schedule entries.
  - Export the schedule API and add deterministic tests.
- Out of scope:
  - Executing a full multiresolution pyramid.
  - Old-core `recon.multires` migration.
  - Alternating solver integration.
- Deep module owner: `tomojax.recon`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Add typed v2 reference FISTA schedule API.
- [x] Export the public schedule API.
- [x] Add deterministic schedule tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the schedule slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/recon/_schedule_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/recon/_schedule_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/recon/_schedule_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 139 tests.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: schedule entries describe v2 reference FISTA configs and residual
  filter configs but do not yet execute a pyramid.
- Deviation: this is the schedule contract before full multiresolution
  reconstruction orchestration.

### Risks

- Risk: callers could expect the schedule API to run reconstruction.
- Mitigation: name it as a schedule resolver and leave execution to a later
  orchestration milestone.

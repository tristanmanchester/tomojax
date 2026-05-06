# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 2 — JAX reference forward model and residual loss
- Goal: add the level-aware projection-domain residual filters required by the
  canonical forward/residual spec.

### Scope

- In scope:
  - Add typed residual filter configuration/result values.
  - Implement raw, low-pass, and band-pass residual filtering for projection
    stacks.
  - Keep filtering in `tomojax.forward`.
  - Add deterministic filter tests and public facade exports.
- Out of scope:
  - Full physical projector geometry.
  - Laminography parameterisations.
  - Geometry Jacobian checks for unsupported DOFs.
- Deep module owner: `tomojax.forward`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Add residual filter implementation.
- [x] Export the public filter API.
- [x] Add deterministic raw/low-pass/band-pass tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the residual-filter slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/forward/_filters.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward/_filters.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/forward/_filters.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py -q`
  passed: 13 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 134 tests.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: use a small separable smoothing kernel as the JAX reference
  low-pass filter and define band-pass as `raw - low_pass(raw)`.
- Deviation: this is not a full multiresolution frequency-domain filter bank;
  it is the first deterministic reference contract for residual filtering.

### Risks

- Risk: tests could over-constrain numerical constants for a future physical
  filter bank.
- Mitigation: test shape, energy-routing, identity behavior, and mask-safe
  contracts instead of exact spectral design details.

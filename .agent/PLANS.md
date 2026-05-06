# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 3 — Reconstruction step: Huber-TV/FISTA preview
- Goal: add the first v2 JAX reference FISTA reconstruction preview against the
  current reference projector.

### Scope

- In scope:
  - Add typed reference FISTA config/result/trace rows.
  - Use the JAX reference projector and masked robust residual loss.
  - Add smoothed TV regularisation, warm start, and non-negativity projection.
  - Add CSV trace artifact export.
  - Add deterministic tiny reconstruction tests.
- Out of scope:
  - Physical projector/backprojector completeness.
  - Multiresolution reconstruction schedules.
  - Production step-size estimation or line search.
- Deep module owner: `tomojax.recon`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Add reference FISTA implementation.
- [x] Export the public FISTA preview API.
- [x] Add deterministic reconstruction and trace tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [x] Commit the FISTA preview slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/recon/_fista_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/recon/_fista_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/recon/_fista_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_reference_fista.py tests/test_v2_module_skeleton.py -q`
  passed: 4 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 136 tests.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: use a fixed configured step size for this first reference FISTA
  slice.
- Deviation: the current projector is still the minimal differentiable
  reference projector, so this is a preview FISTA contract rather than final
  reconstruction quality.

### Risks

- Risk: fixed step size can be problem-dependent.
- Mitigation: expose it in the config and test only tiny deterministic
  reference cases.

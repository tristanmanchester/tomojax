# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 2 — JAX reference forward model and residual loss
- Goal: add a minimal JAX reference forward/residual slice for tiny smoke tests.

### Scope

- In scope:
  - Implement a small parallel-beam JAX reference projector for cubic volumes.
  - Support per-view detector shifts through `PoseParameters.dx_px/dz_px`.
  - Implement masked whitened residuals.
  - Implement pseudo-Huber loss and IRLS weights.
  - Add deterministic smoke tests for projection shape, shift effect, masking,
    robust loss, and weight behavior.
- Out of scope:
  - Full physical tomography/laminography ray geometry.
  - Detector roll, axis rotations, theta scale, and finite-difference geometry
    Jacobian tests.
  - Reconstruction and alignment orchestration.
- Deep module owner: `tomojax.forward`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Inspect current `tomojax.forward` skeleton and loss spec.
- [x] Add minimal JAX reference projector.
- [x] Add masked pseudo-Huber residual/loss helpers.
- [x] Add forward/residual tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the forward/residual slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/forward tests/test_forward_reference.py
  tests/test_v2_module_skeleton.py` passes.
- `uv run basedpyright src/tomojax/forward tests/test_forward_reference.py
  tests/test_v2_module_skeleton.py` passes with 0 errors and 0 warnings.
- `uv run pytest tests/test_forward_reference.py tests/test_v2_module_skeleton.py
  -q` passes with 7 tests.
- `uv run ruff format --check src/tomojax/forward
  tests/test_forward_reference.py tests/test_v2_module_skeleton.py` passes.
- `just imports` passes.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py
  tests/test_align_checkpoint.py tests/test_axes_io.py
  tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py
  tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py
  tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py
  tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py
  tests/test_geometry_gauges.py tests/test_geometry_serialization.py
  tests/test_forward_reference.py -q` passes with 121 tests.
- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup log.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep this as a tiny JAX reference slice rather than adapting old
  `tomojax.core.projector`.
- Deviation: this first projector handles detector shifts and coarse theta
  rotation via array operations only; full ray geometry remains in later Phase 2
  work.

### Risks

- Risk: the smoke projector may be mistaken for the final projector.
- Mitigation: document it as minimal reference scaffolding and keep tests scoped
  to contracts it actually implements.

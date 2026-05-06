# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 1 — Geometry graph, parameters, and gauges
- Goal: add the first typed v2 geometry state and gauge canonicalisation API.

### Scope

- In scope:
  - Add typed setup parameters and per-view 5-DOF pose containers.
  - Add `GeometryState` for setup plus pose arrays.
  - Add gauge canonicalisation transferring mean pose residuals into setup
    parameters with a report.
  - Add tests for zero-centering, inactive `det_v`, and realised-gauge
    preservation by comparing setup-plus-pose totals.
- Out of scope:
  - JSON/CSV geometry artifact writers.
  - Geometry optimiser, Jacobians, Schur solve, or projector integration.
  - Replacing old `tomojax.core.geometry` primitives.
- Deep module owner: `tomojax.geometry`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Inspect current `tomojax.geometry` public facade.
- [x] Add typed geometry state and gauge implementation.
- [x] Add geometry gauge tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the geometry state slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/geometry tests/test_geometry_gauges.py
  tests/test_v2_module_skeleton.py` passes.
- `uv run basedpyright src/tomojax/geometry tests/test_geometry_gauges.py
  tests/test_v2_module_skeleton.py` passes with 0 errors and 0 warnings.
- `uv run pytest tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py
  -q` passes with 5 tests.
- `uv run ruff format --check src/tomojax/geometry
  tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py` passes.
- `just imports` passes.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py
  tests/test_align_checkpoint.py tests/test_axes_io.py
  tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py
  tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py
  tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py
  tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py
  tests/test_geometry_gauges.py -q` passes with 113 tests.
- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup log.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: implement only state/gauge data structures first. Optimisation and
  artifact serialisation remain separate milestones.
- Decision: gauge preservation tests compare realised `setup + residual`
  channels for `det_u/dx`, `theta_offset/phi_residual`, and active `det_v/dz`.

### Risks

- Risk: this initial state model may overlap with old `tomojax.core.geometry`
  names.
- Mitigation: keep v2 types in top-level `tomojax.geometry` and avoid changing
  old core geometry imports in this slice.

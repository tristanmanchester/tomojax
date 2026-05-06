# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 1 — Geometry graph, parameters, and gauges
- Goal: add geometry artifact serialization for the typed v2 geometry state.

### Scope

- In scope:
  - Serialize `GeometryState` setup parameters to `geometry_initial.json` and
    `geometry_final.json`-compatible JSON payloads.
  - Serialize and read per-view 5-DOF pose parameters as `pose_params.csv`.
  - Serialize a simple `pose_decomposition.csv` with setup-plus-pose realised
    channels.
  - Add round-trip tests for JSON and CSV artifacts.
- Out of scope:
  - Geometry optimiser, Jacobians, Schur solve, or projector integration.
  - Full run-directory artifact indexing.
  - Replacing old `tomojax.core.geometry` primitives.
- Deep module owner: `tomojax.geometry`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Inspect current `tomojax.geometry` state/gauge API.
- [x] Add geometry JSON/CSV serialization implementation.
- [x] Add serialization round-trip tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the geometry serialization slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/geometry tests/test_geometry_serialization.py
  tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py` passes.
- `uv run basedpyright src/tomojax/geometry
  tests/test_geometry_serialization.py tests/test_geometry_gauges.py
  tests/test_v2_module_skeleton.py` passes with 0 errors and 0 warnings.
- `uv run pytest tests/test_geometry_serialization.py
  tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py -q` passes
  with 8 tests.
- `uv run ruff format --check src/tomojax/geometry
  tests/test_geometry_serialization.py tests/test_geometry_gauges.py
  tests/test_v2_module_skeleton.py` passes.
- `just imports` passes.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py
  tests/test_align_checkpoint.py tests/test_axes_io.py
  tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py
  tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py
  tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py
  tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py
  tests/test_geometry_gauges.py tests/test_geometry_serialization.py -q`
  passes with 116 tests.
- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup log.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep artifact serialization in `tomojax.geometry` because these
  files are geometry-state contracts, while low-level JSON normalization remains
  in `tomojax.io`.
- Decision: write plain JSON/CSV files first; run-level artifact indexes remain
  a later `tomojax.verify` responsibility.

### Risks

- Risk: JSON schema may need to grow when optimiser observability metadata lands.
- Mitigation: use explicit `schema_version` fields and preserve unknown-free
  typed round-trips for the current Phase 1 state.

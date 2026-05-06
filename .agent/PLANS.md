# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 2 differentiability bridge before pose optimisation
- Goal: make detector-shift projection differentiable for future pose solvers.

### Scope

- In scope:
  - Replace rounded detector shifts in the minimal forward projector with
    periodic linear interpolation.
  - Add an array-based projection helper that accepts JAX pose/setup shift arrays.
  - Add tests for fractional detector shifts and differentiability with respect
    to `dx_px`.
- Out of scope:
  - Pose-only LM/GN optimisation.
  - Detector roll, axis rotations, theta scale, and laminography ray geometry.
  - Reconstruction or alignment API changes.
- Deep module owner: `tomojax.forward`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Inspect current minimal forward projector.
- [x] Add differentiable periodic detector shift helper.
- [x] Add JAX array-based projection helper.
- [x] Add differentiability tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the differentiable-shift slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/forward tests/test_forward_reference.py
  tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py` passes.
- `uv run basedpyright src/tomojax/forward tests/test_forward_reference.py
  tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py` passes with
  0 errors and 0 warnings.
- `uv run pytest tests/test_forward_reference.py tests/test_vertical_smoke.py
  tests/test_v2_module_skeleton.py -q` passes with 12 tests.
- `uv run ruff format --check src/tomojax/forward
  tests/test_forward_reference.py tests/test_vertical_smoke.py
  tests/test_v2_module_skeleton.py` passes.
- `just imports` passes.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py
  tests/test_align_checkpoint.py tests/test_axes_io.py
  tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py
  tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py
  tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py
  tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py
  tests/test_geometry_gauges.py tests/test_geometry_serialization.py
  tests/test_forward_reference.py tests/test_vertical_smoke.py -q` passes with
  126 tests.
- `uv run python` without `JAX_PLATFORM_NAME=cpu` reports a CUDA plugin warning
  about missing cuSPARSE, then falls back to CPU. Tests still run successfully
  under `uv run`.
- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup log.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep the current coarse theta handling, but remove non-differentiable
  rounded detector shifts.
- Deviation: this still is not full physical ray geometry.

### Risks

- Risk: periodic boundary interpolation is a smoke-model simplification.
- Mitigation: document this as reference scaffolding and keep physical projector
  expansion as future Phase 2 work.

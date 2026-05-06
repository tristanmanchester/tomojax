# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Minimal vertical-slice bridge across Phase 2 and Phase 3
- Goal: add a tiny reconstruction/alignment smoke path using the new v2
  geometry, forward, robust residual, and gauge APIs.

### Scope

- In scope:
  - Make the minimal forward projector respect setup `det_u_px` and active
    `det_v_px` so gauge canonicalisation preserves projections.
  - Add a simple reference backprojection/preview reconstruction helper in
    `tomojax.recon`.
  - Add an alignment smoke report in `tomojax.align` that reconstructs a preview
    volume, computes masked robust projection loss, canonicalises gauges, and
    verifies the canonical projection loss is preserved.
  - Add tests for gauge-equivalent projection preservation and the smoke report.
- Out of scope:
  - FISTA/Huber-TV implementation.
  - LM/GN geometry optimisation.
  - Full physical projector support for detector roll, axis rotations, or
    laminography.
- Deep module owners: `tomojax.forward`, `tomojax.recon`, and `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Inspect current `recon` and `align` public facades.
- [x] Update minimal forward projector for setup detector shifts.
- [x] Add reference reconstruction smoke helper.
- [x] Add alignment smoke report.
- [x] Add tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the smoke vertical slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/forward src/tomojax/recon/_reference.py
  src/tomojax/recon/api.py src/tomojax/recon/__init__.py
  src/tomojax/align/_smoke.py src/tomojax/align/api.py
  src/tomojax/align/__init__.py tests/test_vertical_smoke.py
  tests/test_forward_reference.py tests/test_v2_module_skeleton.py` passes.
- `uv run basedpyright src/tomojax/forward src/tomojax/recon/_reference.py
  src/tomojax/recon/api.py src/tomojax/recon/__init__.py
  src/tomojax/align/_smoke.py src/tomojax/align/api.py
  src/tomojax/align/__init__.py tests/test_vertical_smoke.py
  tests/test_forward_reference.py tests/test_v2_module_skeleton.py` passes with
  0 errors and 0 warnings.
- `uv run pytest tests/test_vertical_smoke.py tests/test_forward_reference.py
  tests/test_v2_module_skeleton.py -q` passes with 10 tests.
- `uv run ruff format --check src/tomojax/forward
  src/tomojax/recon/_reference.py src/tomojax/recon/api.py
  src/tomojax/recon/__init__.py src/tomojax/align/_smoke.py
  src/tomojax/align/api.py src/tomojax/align/__init__.py
  tests/test_vertical_smoke.py tests/test_forward_reference.py
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
  124 tests.
- A broad `uv run ruff format --check src/tomojax/forward src/tomojax/recon
  src/tomojax/align ...` still reports 20 untouched transitional align/recon
  files that would be reformatted. This is outside this slice and remains part
  of the broader legacy cleanup.
- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup log.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: this smoke path is deliberately small and deterministic. It proves
  module wiring and artifact-style reporting, not final numerical performance.
- Deviation: reconstruction preview is average backprojection, not FISTA/TV.
  The full reconstruction milestone remains open.

### Risks

- Risk: smoke helpers can become accidental product APIs.
- Mitigation: README/log must label them as reference smoke helpers and future
  milestones should replace them with FISTA/LM defaults.

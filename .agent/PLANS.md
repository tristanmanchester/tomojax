# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 0 dataset/benchmark foundation, scoped to deterministic
  synthetic dataset artifacts
- Goal: add a typed `tomojax.datasets` foundation that can load the v2
  synthetic benchmark manifest and emit deterministic smoke artifacts.

### Scope

- In scope:
  - Load the five synthetic benchmark specs from
    `docs/tomojax-v2/benchmark_manifest.yaml`.
  - Generate a deterministic procedural phantom in 32^3 smoke mode and 128^3
    configured mode.
  - Write dataset artifacts: manifest, volume, projections, mask, nominal/true
    geometry, pose CSV, motion CSV, nuisance/noise JSON, and recovery
    tolerances.
  - Add tests for determinism, artifact presence, and manifest content.
- Out of scope:
  - Final differentiable JAX projector correctness.
  - Full physical laminography projection fidelity.
  - Geometry optimisation or reconstruction.
  - Current TomoJAX benchmark runner/report generation.
- Deep module owner: `tomojax.datasets`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`
- `docs/tomojax-v2/benchmark_manifest.yaml`

### Tasks

- [x] Inspect benchmark manifest and existing transitional phantom/simulate code.
- [x] Add typed dataset spec/loading API.
- [x] Add deterministic phantom/projection artifact writer.
- [x] Add dataset tests.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the dataset foundation slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/datasets tests/test_synthetic_datasets.py
  tests/test_v2_module_skeleton.py` passes.
- `uv run basedpyright src/tomojax/datasets tests/test_synthetic_datasets.py
  tests/test_v2_module_skeleton.py` passes with 0 errors and 0 warnings.
- `uv run pytest tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py
  -q` passes with 5 tests.
- `uv run ruff format --check src/tomojax/datasets
  tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py` passes.
- `just imports` passes.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py
  tests/test_align_checkpoint.py tests/test_axes_io.py
  tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py
  tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py
  tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py
  tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py -q`
  passes with 110 tests.
- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the previous milestone log.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: use `tomojax.datasets` as the owner for v2 synthetic benchmark
  artifacts instead of extending old `tomojax.data`.
- Decision: the first projection writer may be a deterministic CPU smoke
  projector. It is not the final differentiable `tomojax.forward` reference
  path.
- Deviation: 128^3 mode is configured but not exercised by default tests to keep
  pre-commit fast.

### Risks

- Risk: simple smoke projections could be mistaken for the final forward model.
- Mitigation: name and docs should describe them as dataset foundation
  artifacts only; the JAX reference projector milestone remains separate.

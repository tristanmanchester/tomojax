# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: expose lightweight generated synthetic array metadata through the
  public dataset loader.

### Scope

- In scope:
  - Resolve volume, projection, and mask arrays from the manifest artifact map.
  - Validate their shapes and dtypes with memory-mapped NumPy loads.
  - Expose array paths and metadata on `SyntheticDatasetSidecars`.
  - Add focused tests for array metadata and missing array entries.
- Out of scope:
  - Alternating solver ingestion of generated benchmark projections.
  - Stripe/ring bias fields.
  - Changing align-auto defaults.
  - Removing existing manifest-spec geometry artifacts.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.datasets`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add loader array metadata fields.
- [x] Validate volume/projection/mask array shape and dtype.
- [x] Add focused loader tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the synthetic array metadata loader slice.

### Validation

- `uv run ruff format src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed: 4 files left unchanged after import-order fixes.
- `uv run ruff check src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py -q`
  passed: 11 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep array readback metadata-only and memory-mapped. Do not route generated
  arrays into the smoke solver in this slice.

### Risks

- Risk: generated projections still come from the NumPy smoke projector and are
  not solver inputs.
- Mitigation: loader exposes shapes, dtypes, and paths only; solver ingestion
  remains a later compatibility milestone.

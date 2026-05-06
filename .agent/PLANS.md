# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: expose manifest-indexed synthetic v2 geometry sidecars through the
  public dataset API.

### Scope

- In scope:
  - Add a typed public loader for generated synthetic dataset artifacts.
  - Resolve v2 geometry/pose sidecars from `dataset_manifest.json`.
  - Read nominal, corrupted, and true sidecars through public geometry APIs.
  - Add focused tests for loader readback and missing artifact-map errors.
- Out of scope:
  - Alternating solver ingestion of generated benchmark projections.
  - Stripe/ring bias fields.
  - Changing align-auto defaults.
  - Removing existing manifest-spec geometry artifacts.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.datasets`, consuming public `tomojax.geometry`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add public synthetic artifact loader.
- [x] Export loader through dataset facade and README.
- [x] Add focused loader tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the synthetic sidecar loader slice.

### Validation

- `uv run ruff format src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed: 4 files left unchanged after focused fixes.
- `uv run ruff check src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py -q`
  passed: 10 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep the loader data-only. It should not run alignment, reconstruction, or
  projection compatibility checks.

### Risks

- Risk: generated projections still come from the NumPy smoke projector.
- Mitigation: loader exposes artifacts and v2 geometry states only; solver
  ingestion remains an explicit later milestone.

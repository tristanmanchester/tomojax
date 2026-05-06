# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: advertise v2 synthetic benchmark geometry sidecars in the dataset
  manifest.

### Scope

- In scope:
  - Add a stable artifact map to `dataset_manifest.json`.
  - Include v2 geometry and pose sidecars in that map.
  - Keep existing artifact filenames unchanged.
  - Add focused tests that resolve sidecar paths from the manifest and read them
    through public `tomojax.geometry` APIs.
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

- [x] Add manifest artifact map.
- [x] Include v2 sidecar path entries.
- [x] Add focused manifest-discovery readback tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the manifest sidecar index slice.

### Validation

- `uv run ruff format src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed: 1 file reformatted, 1 file left unchanged.
- `uv run ruff check src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py -q`
  passed: 6 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Add manifest discovery metadata without renaming or removing existing
  synthetic benchmark artifacts.

### Risks

- Risk: manifest sidecar discovery still does not imply solver ingestion.
- Mitigation: tests only assert path discoverability and public geometry
  readback.

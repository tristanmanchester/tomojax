# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: add loader-level synthetic array/geometry consistency checks.

### Scope

- In scope:
  - Compare loaded array metadata against `dataset_manifest.json`.
  - Check projection/mask shapes match and view counts match loaded geometry.
  - Expose a compact consistency payload on `SyntheticDatasetSidecars`.
  - Add focused loader tests for passing and failing consistency.
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

- [x] Add loader consistency payload.
- [x] Check manifest shape/view metadata against arrays and geometry.
- [x] Add focused loader tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the synthetic consistency loader slice.

### Validation

- `uv run ruff format src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed: 4 files left unchanged.
- `uv run ruff check src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py -q`
  passed: 12 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep consistency checks structural only. Do not compare generated projections
  numerically against the JAX reference projector in this slice.

### Risks

- Risk: structural consistency still does not prove physical projector
  compatibility.
- Mitigation: payload should report shape/view consistency only; reference
  projector compatibility remains later work.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: make synthetic benchmark nuisance artifacts exercise the owned
  background-gradient model.

### Scope

- In scope:
  - Realize `background_drift = low_frequency_vertical_gradient` as a per-view
    vertical-gradient background field in synthetic projections.
  - Record the gradient coefficients in `nuisance_truth.json`.
  - Add focused dataset tests proving clean projections skip but record the
    nuisance, while dirty projections apply the vertical field.
- Out of scope:
  - Alternating solver ingestion of generated benchmark projections.
  - Stripe/ring bias fields.
  - Changing align-auto defaults.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.datasets`, using the public nuisance contract
  shape already implemented in `tomojax.nuisance`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add vertical background-gradient realization to synthetic writer.
- [x] Add focused synthetic dataset tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the synthetic background-gradient nuisance slice.

### Validation

- `uv run ruff format src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed: 1 file reformatted, 1 file left unchanged.
- `uv run ruff check src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_nuisance_background.py -q`
  passed: 10 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep the dataset writer NumPy-only and deterministic; this slice should not
  introduce solver coupling.

### Risks

- Risk: generated benchmark datasets still are not consumed by the alternating
  solver path.
- Mitigation: this slice fixes the nuisance artifact truth/apply contract first
  so later ingestion has meaningful nuisance-bearing projections.

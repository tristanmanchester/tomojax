# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: apply deterministic nuisance drift in synthetic benchmark artifacts.

### Scope

- In scope:
  - Apply per-view gain drift and simple background offsets in
    `generate_synthetic_dataset`.
  - Record realized gain/offset arrays in `nuisance_truth.json`.
  - Keep `clean=True` behavior nuisance-free.
  - Add focused synthetic dataset tests for nuisance-bearing specs.
- Out of scope:
  - Hot/dead pixels, stripes, bad views, and partial-FOV masks.
  - Loading generated benchmark projections into the alternating solver.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.datasets`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add deterministic gain/background realization helpers.
- [x] Apply nuisance when `clean=False`.
- [x] Record realized nuisance truth payload.
- [x] Add focused synthetic dataset tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the synthetic nuisance slice.

### Validation

- `uv run ruff format src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `uv run ruff check src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_nuisance_gain_offset.py -q`
  passed: 9 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Implement only gain and scalar per-view background offsets first, matching the
  current Phase 8 gain/offset model. More complex nuisance terms remain recorded
  as spec metadata until they have owned models.

### Risks

- Risk: this does not yet apply hot/dead pixels, stripes, partial-FOV masks, or
  bad views from the hardest synthetic specs.
- Mitigation: record unsupported terms in the nuisance truth payload so later
  slices can add them explicitly.

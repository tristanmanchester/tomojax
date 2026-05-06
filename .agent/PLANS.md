# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: emit `GeometryState`-compatible synthetic benchmark geometry sidecars.

### Scope

- In scope:
  - Add v2 `GeometryState` JSON sidecars for nominal, corrupted, and true
    synthetic benchmark geometry.
  - Add radian `pose_params.csv` sidecars matching the public geometry reader.
  - Keep the existing manifest-spec geometry JSON and degree pose CSV files for
    benchmark contract continuity.
  - Add focused dataset tests proving the new geometry sidecars can be read
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

- [x] Add v2 geometry state paths and writes.
- [x] Add pose params CSV sidecars in radians.
- [x] Add focused public geometry readback tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the v2 geometry sidecar slice.

### Validation

- `uv run ruff format src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep existing synthetic benchmark artifacts stable; add v2 sidecars rather
  than changing the manifest-defined JSON schema in place.

### Risks

- Risk: generated benchmark projections still use the NumPy smoke projector
  rather than the JAX reference forward model.
- Mitigation: this slice only creates a public geometry-format bridge for a
  later ingestion step.

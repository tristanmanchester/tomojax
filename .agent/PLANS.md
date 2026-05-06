# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: make `align-auto` validate generated synthetic v2 sidecars through the
  public dataset loader.

### Scope

- In scope:
  - Call `load_synthetic_dataset_sidecars` after generating a named synthetic
    benchmark sidecar dataset.
  - Record a compact readback summary in the smoke synthetic dataset payload.
  - Keep the sidecar data out of the alternating solver update path.
  - Add focused CLI tests for the summary.
- Out of scope:
  - Alternating solver ingestion of generated benchmark projections.
  - Stripe/ring bias fields.
  - Changing align-auto defaults.
  - Removing existing manifest-spec geometry artifacts.
  - Further legacy Ruff cleanup.
- Deep module owners: `tomojax.cli`, `tomojax.align`, and public
  `tomojax.datasets` loader consumption.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Validate generated synthetic sidecars in `align-auto`.
- [x] Record readback summary in verification/manifest/config artifacts.
- [x] Add focused CLI tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the align-auto sidecar validation slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed: 1 file reformatted, 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py tests/test_synthetic_datasets.py -q`
  passed: 14 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep sidecar validation as metadata/readback only. Do not route generated
  projections into the smoke solver in this slice.

### Risks

- Risk: generated projections still come from the NumPy smoke projector and are
  not solver inputs.
- Mitigation: artifact payloads must label this as sidecar readback, not solver
  ingestion.

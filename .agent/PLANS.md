# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: record synthetic loader consistency in `align-auto` sidecar readback
  artifacts.

### Scope

- In scope:
  - Include `SyntheticDatasetSidecars.consistency` in `align-auto`
    `sidecar_readback`.
  - Record compact consistency status in `config_resolved.toml`.
  - Add focused CLI assertions for the consistency payload.
- Out of scope:
  - Alternating solver ingestion of generated benchmark projections.
  - Stripe/ring bias fields.
  - Changing align-auto defaults.
  - Removing existing manifest-spec geometry artifacts.
  - Further legacy Ruff cleanup.
- Deep module owners: `tomojax.cli` and `tomojax.align` artifact plumbing.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add consistency payload to align-auto sidecar readback.
- [x] Add resolved-config consistency field.
- [x] Add focused CLI tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the align-auto consistency readback slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed: 1 file reformatted, 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py tests/test_synthetic_datasets.py -q`
  passed: 16 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep the payload under `sidecar_readback`; do not change solver inputs.

### Risks

- Risk: consistency can be mistaken for solver ingestion.
- Mitigation: label it as sidecar readback only and keep generated arrays out of
  the smoke solver.

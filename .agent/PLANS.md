# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: let `align-auto` generate nuisance-applied synthetic benchmark artifacts
  on request.

### Scope

- In scope:
  - Record whether synthetic nuisance terms were applied to generated
    projections in `nuisance_truth.json`.
  - Add an `align-auto` flag to generate dirty synthetic benchmark projections.
  - Record the applied/clean choice in smoke verification, manifest, and
    resolved config artifacts.
  - Add focused CLI/dataset tests for the default clean path and opt-in dirty
    path.
- Out of scope:
  - Alternating solver ingestion of generated benchmark projections.
  - Stripe/ring bias fields.
  - Changing align-auto defaults.
  - Further legacy Ruff cleanup.
- Deep module owners: `tomojax.datasets`, `tomojax.cli`, and `tomojax.align`
  artifact plumbing.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add synthetic nuisance applied/clean truth field.
- [x] Add `align-auto --apply-synthetic-nuisance`.
- [x] Record choice in run artifacts.
- [x] Add focused CLI/dataset tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the align-auto dirty synthetic artifact slice.

### Validation

- `uv run ruff format src/tomojax/datasets/_writer.py src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py`
  passed: 5 files left unchanged.
- `uv run ruff check src/tomojax/datasets/_writer.py src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_writer.py src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_align_auto_cli.py -q`
  passed: 12 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep the default synthetic benchmark generation clean. Dirty projections must
  be explicit until the alternating solver can consume generated benchmark
  geometry/projections through a compatible reference path.

### Risks

- Risk: generated benchmark datasets still are metadata/artifact sidecars for
  `align-auto`.
- Mitigation: this slice makes clean versus nuisance-applied sidecars explicit
  and testable, without pretending the solver has ingested them.

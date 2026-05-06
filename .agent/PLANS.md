# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Alternating solver and continuation
- Goal: run Phase 7 auto alignment from a named synthetic benchmark spec.

### Scope

- In scope:
  - Add a focused `align-auto` option for a named synthetic128 benchmark spec.
  - Generate deterministic synthetic dataset artifacts for that spec before the
    Phase 7 auto run.
  - Thread the benchmark dataset identity into resolved config, manifest, and
    verification artifacts.
  - Add CLI and smoke tests for the benchmark-facing path.
- Out of scope:
  - Further legacy Ruff cleanup.
  - GPU/Pallas fast paths.
  - Full production dataset loading through the new command.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add typed benchmark dataset metadata to the Phase 7 smoke config.
- [x] Generate named synthetic dataset artifacts from `align-auto`.
- [x] Record dataset metadata in config, manifest, and verification artifacts.
- [x] Add focused CLI and smoke tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the benchmark-facing auto path slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py tests/test_synthetic_datasets.py tests/test_verify_artifacts.py -q`
  passed: 12 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep the solver input as the current deterministic Phase 7 smoke fixture for
  this slice; the named synthetic dataset artifacts are generated and recorded
  as benchmark context until external dataset ingestion is implemented.

### Risks

- Risk: this is benchmark-context generation, not full external dataset
  ingestion.
- Mitigation: record the limitation in the implementation log and keep the
  command/test scope explicit.

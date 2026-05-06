# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: expose the existing geometry-update volume source through `align-auto`.

### Scope

- In scope:
  - Add an `align-auto` CLI option for the existing
    `AlternatingSmokeConfig.geometry_update_volume_source` setting.
  - Cover CLI propagation into run artifacts.
  - Record focused validation.
- Out of scope:
  - Adding or changing artifact/report/observability fields.
  - New benchmark ingestion behavior.
  - Changing the default stopped-reconstruction solver path.
  - Solver tuning beyond exposing the existing config.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.cli` entrypoint over public `tomojax.align.api`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add CLI option for geometry-update volume source.
- [x] Add focused CLI propagation coverage.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the CLI source option slice.

### Validation

- `uv run ruff format src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep default `stopped_reconstruction`; use the new flag only for explicit
  oracle/diagnostic benchmark runs.

### Risks

- Risk: users may misuse `fixed_synthetic_truth` as a production mode.
- Mitigation: CLI help labels it as an explicit geometry-update volume source;
  default remains `stopped_reconstruction`.

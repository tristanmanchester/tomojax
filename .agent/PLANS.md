# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: split the alternating solver private implementation before adding more
  benchmark-ingestion behavior.

### Scope

- In scope:
  - Keep `tomojax.align._alternating` as the public-compatible facade.
  - Move the alternating loop into a private orchestration implementation.
  - Move geometry-update and per-level smoke helpers into cohesive private files.
  - Preserve existing artifact, verification/report payload, held-out check, and
    config/result helper boundaries.
- Out of scope:
  - New benchmark ingestion behavior.
  - Artifact-shape polishing.
  - Public API changes.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Split alternating orchestration from the public facade.
- [x] Move geometry-update helpers out of the facade.
- [x] Move level-summary/time-state helpers out of the facade.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the align-module cleanup slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_level_helpers.py`
  passed: 4 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_level_helpers.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_level_helpers.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- `_alternating.py` remains importable so the existing public facade in
  `tomojax.align.api` does not change.

### Risks

- Risk: moving private helpers could accidentally change smoke-run behavior.
- Mitigation: preserve existing tests and add focused import/API coverage if
  needed before running the 32^3 smoke tests.

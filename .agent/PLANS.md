# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Synthetic benchmark foundation / Phase 7 smoke artifacts
- Goal: add per-view projection residual metrics to the Phase 7 deterministic
  smoke run.

### Scope

- In scope:
  - Expand `residual_metrics.csv` with per-view raw projection residual rows.
  - Include RMSE, MAE, robust loss, valid-pixel fraction, and raw RMSE fields.
  - Keep existing per-level summary rows in the same artifact.
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

- [x] Add per-view residual metric rows.
- [x] Extend focused residual metric tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the residual metrics expansion slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 6 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Add a `row_type` column so per-level summary rows and per-view residual rows
  can coexist without ambiguity.

### Risks

- Risk: per-view metrics are raw residual only in this slice.
- Mitigation: use contract field names and leave filtered low-pass/band-pass
  view metrics for a later pass.

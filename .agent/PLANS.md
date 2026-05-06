# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Synthetic benchmark foundation / Phase 7 smoke artifacts
- Goal: record final volume recovery metrics in the Phase 7 deterministic
  smoke verification report.

### Scope

- In scope:
  - Compute final-vs-truth volume RMSE, MAE, and NMSE.
  - Compare volume NMSE against a smoke-specific recovery tolerance.
  - Record the metrics and pass/fail flags in `verification.json` and
    `recovery_tolerances.json`.
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

- [x] Add volume recovery tolerances.
- [x] Compute final volume recovery metrics.
- [x] Extend focused smoke tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the smoke volume metrics slice.

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

- Keep the tolerance loose enough for the one-iteration smoke FISTA path while
  still recording the metric needed by the benchmark contract.

### Risks

- Risk: one-iteration smoke reconstruction is not intended to produce high
  volume quality.
- Mitigation: record metrics for audit now and leave stricter benchmark
  thresholds for larger schedules.

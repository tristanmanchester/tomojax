# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Synthetic benchmark foundation / Phase 7 smoke artifacts
- Goal: persist a deterministic geometry trace for the Phase 7 smoke bundle.

### Scope

- In scope:
  - Write `geometry_trace.csv` from the existing per-level geometry update
    summaries.
  - Include level, role, attempted/executed updates, loss delta, gauge/update
    predicates, skip state, and early-exit reason.
  - Index the trace artifact and extend focused smoke tests.
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

- [x] Add `geometry_trace.csv` artifact path.
- [x] Write deterministic per-level geometry trace rows.
- [x] Extend smoke artifact/index tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the geometry trace artifact slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Derive geometry trace rows from `AlternatingLevelSummary` in this slice,
  rather than introducing a second per-update trace model before real LM/GN
  updates exist.

### Risks

- Risk: geometry trace granularity is per level, not per inner LM/GN step.
- Mitigation: keep the schema explicit and refine granularity when real
  geometry optimiser steps replace the smoke canonicalisation update.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 — alternating solver and continuation
- Goal: add explicit early-exit continuation behavior to the deterministic
  32^3 alternating smoke run.

### Scope

- In scope:
  - Add a conditional level-2 continuation entry to the `smoke32` schedule.
  - Skip level 2 when coarse verification passes.
  - Model level-1 geometry as verification-triggered and skipped by default
    when coarse verification passes.
  - Record skipped levels/geometry in summaries and verification artifacts.
- Out of scope:
  - Further legacy Ruff cleanup.
  - Full Phase 7 production profile presets.
  - CLI integration.
  - GPU/Pallas fast paths.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Extend the continuation schedule with conditional level 2 and level-1
      geometry metadata.
- [x] Implement skipped-level and skipped-geometry summaries.
- [x] Record early-exit decisions in verification artifacts.
- [x] Extend focused tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Phase 7 early-exit slice.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_vertical_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Continue to use gauge canonicalisation as the only geometry update in this
  smoke profile; this slice only changes continuation control flow.
- A skipped level should still appear in `alignment_summary.csv`,
  `residual_metrics.csv`, and `verification.json` so early-exit behavior is
  auditable.

### Risks

- Risk: skipped-level rows can be mistaken for executed reconstructions.
- Mitigation: add an explicit `skipped_level` field to level summaries and
  artifacts.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 — alternating solver and continuation
- Goal: add Phase 7 continuation profile presets.

### Scope

- In scope:
  - Add `lightning`, `balanced`, and `reference` continuation schedules.
  - Keep `smoke32` as the deterministic test profile.
  - Test profile level factors, ordering, conditional level-2 behavior, and
    stricter reference iterations/geometry updates.
- Out of scope:
  - Further legacy Ruff cleanup.
  - CLI integration.
  - GPU/Pallas fast paths.
  - Running the heavier production presets through the smoke test.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add continuation profile schedules.
- [x] Add focused schedule contract tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Phase 7 profile slice.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/api.py tests/test_continuation_schedules.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/api.py tests/test_continuation_schedules.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py tests/test_continuation_schedules.py`
  passed.
- `uv run pytest tests/test_continuation_schedules.py tests/test_alternating_solver_smoke.py -q`
  passed: 8 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Profile schedules should be deterministic data only; do not start CLI or
  runtime profile plumbing in this slice.
- `reference` should be stricter than `balanced`, and `balanced` should be
  stricter than `lightning` in reconstruction iterations and geometry updates.

### Risks

- Risk: profile schedules are not yet empirically tuned.
- Mitigation: encode conservative, monotonic presets and record tuning as a
  follow-up risk.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 — alternating solver and continuation
- Goal: implement the smallest v2 alternating-solver/continuation vertical slice
  with deterministic 32^3 synthetic artifacts.

### Scope

- In scope:
  - Add a minimal continuation schedule owned by `tomojax.align`.
  - Add a stopped-volume alternating synthetic smoke runner.
  - Emit `alignment_summary.csv`, `verification.json`,
    `artifact_index.json`, `geometry_initial.json`, `geometry_final.json`,
    `pose_params.csv`, `pose_decomposition.csv`, and `fista_trace.csv`.
  - Add a deterministic 32^3 synthetic smoke test.
- Out of scope:
  - Further legacy Ruff cleanup.
  - Full Phase 7 production schedule profiles and adaptive escalation.
  - CLI integration.
  - GPU/Pallas fast paths.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add continuation schedule and alternating smoke runner.
- [x] Write deterministic artifact contract files.
- [x] Add 32^3 synthetic smoke tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Phase 7 vertical slice.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_vertical_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep `tomojax.align.__all__` unchanged because that package facade is still
  contract-tested for the legacy API; expose Phase 7 names from
  `tomojax.align.api`.
- The first vertical slice uses deterministic gauge canonicalisation as the
  geometry update so the smoke run records real geometry artifacts without
  adding the full Schur LM production loop yet.

### Risks

- Risk: 32^3 FISTA smoke runtime can become slow under JAX compilation.
- Mitigation: use a tiny deterministic schedule and focused tests.

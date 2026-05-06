# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 — alternating solver and continuation
- Goal: expand the deterministic 32^3 alternating smoke run to emit the
  core verification/artifact contract scaffold.

### Scope

- In scope:
  - Persist `final_volume.npy` from the 32^3 alternating smoke result.
  - Emit run-level contract scaffolding: `run_manifest.json`,
    `config_resolved.toml`, `input_summary.json`, `projection_stats.json`,
    `mask_summary.json`, `gauge_report.json`, `backend_report.json`, and
    `residual_metrics.csv`.
  - Update artifact index and focused tests to cover the expanded contract.
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

- [x] Add expanded artifact payloads to the alternating smoke runner.
- [x] Persist the final 32^3 volume artifact.
- [x] Extend tests for artifact index and contract files.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Phase 7 artifact contract slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_vertical_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep this slice inside `tomojax.align._alternating`; do not introduce a
  generic artifact helper module.
- Artifact fields can be smoke-profile minimal, but every emitted file must be
  listed in `artifact_index.json`.

### Risks

- Risk: the smoke artifact scaffold may not yet satisfy the full production
  schema.
- Mitigation: make the files explicit, deterministic, and covered by focused
  tests while recording remaining production-schema gaps in the log.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Alternating solver and continuation
- Goal: make the Phase 7 Schur geometry-update volume source explicit.

### Scope

- In scope:
  - Add a typed smoke config field for the Schur geometry-update volume source.
  - Record the selected source in `config_resolved.toml`, `run_manifest.json`,
    `verification.json`, and `schur_diagnostics.json`.
  - Keep the default as the fixed synthetic smoke volume until stopped-latent
    recovery passes.
  - Add focused smoke tests for the explicit source contract.
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

- [x] Add typed geometry update volume-source config.
- [x] Thread the source into Schur update and artifact payloads.
- [x] Extend focused smoke tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the volume-source contract slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Treat fixed synthetic volume as an explicit smoke-only source, not an implicit
  alternating-solver behavior.

### Risks

- Risk: the default source is still smoke-only.
- Mitigation: expose the source in config and artifacts so stopped-latent
  integration can be made and verified as a separate numerical change.

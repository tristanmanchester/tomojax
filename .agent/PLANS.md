# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 — alternating solver and continuation
- Goal: add a one-command Phase 7 auto-alignment smoke entrypoint.

### Scope

- In scope:
  - Add a CLI module for the deterministic `align=auto` smoke pipeline.
  - Add a console script that writes final volume, geometry, and verification
    artifacts into a run directory.
  - Test the command help and a full smoke invocation.
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

- [x] Add CLI command module.
- [x] Add console script entrypoint.
- [x] Add focused CLI tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Phase 7 CLI smoke slice.

### Validation

- `uv run ruff format src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run ruff check src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run pytest tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.
- `uv run tomojax-align-auto-smoke --help` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- The command is smoke-profile-only for now and should clearly say so.
- The command must call `AlternatingAlignmentSolver`, not duplicate solver
  logic in `tomojax.cli`.

### Risks

- Risk: users may mistake the smoke command for full dataset alignment.
- Mitigation: name and help text identify it as the deterministic Phase 7
  smoke entrypoint.

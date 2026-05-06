# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Alternating solver and continuation
- Goal: harden the Phase 7 `align=auto` CLI acceptance contract.

### Scope

- In scope:
  - Assert the CLI command writes a passed verification report.
  - Assert coarse early exit skips level-1 geometry in command-level tests.
  - Assert Schur diagnostics and continuation prior trace fields are emitted by
    the command path.
  - Add focused CLI validation and `just imports`.
- Out of scope:
  - Further legacy Ruff cleanup.
  - GPU/Pallas fast paths.
  - Loading production datasets through `align-auto`.
- Deep module owner: `tomojax.cli`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Expand `align-auto` CLI tests from file existence to acceptance contract.
- [x] Assert emitted verification, summary, geometry trace, and Schur artifacts.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the CLI contract slice.

### Validation

- `uv run ruff format tests/test_align_auto_cli.py` passed.
- `uv run ruff check tests/test_align_auto_cli.py` passed.
- `uv run basedpyright tests/test_align_auto_cli.py` passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 3 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep this as command-path verification of the existing Phase 7 solver rather
  than adding new placeholder reports or dataset loading behavior.

### Risks

- Risk: this is a test-contract slice, not a new numerical capability.
- Mitigation: it guards the canonical Phase 7 acceptance criterion that one
  command produces final volume, geometry, and verification artifacts.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: expose gain/offset nuisance fitting through the alternating smoke CLI.

### Scope

- In scope:
  - Add an alternating smoke config flag for Schur gain/offset fitting.
  - Pass the flag into Schur geometry updates.
  - Add an `align-auto` CLI switch for the opt-in nuisance path.
  - Record the resolved option in config/manifest or diagnostics.
  - Add focused smoke/CLI tests.
- Out of scope:
  - Default-enabling nuisance fitting.
  - Background fields.
  - Weak DOF auto-activation rules.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` and `tomojax.cli`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add alternating config plumbing.
- [x] Add CLI flag and resolved artifact field.
- [x] Add focused smoke/CLI tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the CLI nuisance toggle slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 9 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep nuisance fitting opt-in at the command/config level until benchmark
  datasets with nuisance drift are wired into the solver path.

### Risks

- Risk: the default Phase 7 smoke still runs without nuisance fitting.
- Mitigation: command-level tests will cover the opt-in path and diagnostics.

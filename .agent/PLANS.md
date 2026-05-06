# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: expose background nuisance fitting through the alternating smoke CLI.

### Scope

- In scope:
  - Add an alternating smoke config flag for background nuisance fitting.
  - Thread the option into Schur geometry updates.
  - Add an `align-auto` CLI switch.
  - Record the option in resolved config, manifest, and verification artifacts.
  - Add focused smoke and CLI tests for the default and opt-in paths.
- Out of scope:
  - Default-enabling background fitting in Phase 7 smoke.
  - Stripe/ring bias fields.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`, with CLI plumbing in `tomojax.cli`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add alternating background config plumbing.
- [x] Add CLI flag and resolved artifact fields.
- [x] Add focused smoke/CLI tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the CLI background toggle slice.

### Validation

- `uv run ruff format --check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 12 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep background fitting disabled by default until nuisance-bearing benchmark
  projections are loaded by the solver path.

### Risks

- Risk: the current default smoke has no background drift, so this slice only
  proves the opt-in plumbing and diagnostics.
- Mitigation: benchmark ingestion should exercise the toggle with
  nuisance-bearing data.

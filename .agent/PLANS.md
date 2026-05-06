# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 weak DOF handling
- Goal: wire joint Schur block priors through alternating and `align-auto`.

### Scope

- In scope:
  - Add optional setup/pose prior strengths to `AlternatingSmokeConfig`.
  - Pass the optional block priors into joint Schur geometry updates.
  - Expose explicit `align-auto` flags and cover CLI propagation.
- Out of scope:
  - Adding new verification/report fields beyond resolved config text.
  - New benchmark ingestion behavior.
  - Changing the default stopped-reconstruction solver path.
  - Solver tuning.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` alternating orchestration and
  `tomojax.cli` entrypoint.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add optional setup/pose prior strengths to alternating config.
- [x] Expose explicit `align-auto` flags.
- [x] Add focused propagation coverage.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the alternating/CLI prior slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_artifacts.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed: 1 file reformatted, 5 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_artifacts.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_artifacts.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 12 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Defaults remain unset, so schedule-level shared prior behavior is unchanged.

### Risks

- Risk: users can overtune oracle runs with strong priors.
- Mitigation: expose the knobs explicitly and keep defaults unchanged.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Alternating solver and continuation
- Goal: use the stopped reconstructed latent as the Phase 7 Schur geometry
  update volume.

### Scope

- In scope:
  - Make the default Phase 7 smoke Schur update use the stopped reconstructed
    latent volume.
  - Adjust only the deterministic smoke geometry/update budget needed for
    stopped-latent supported DOF recovery.
  - Keep the fixed synthetic truth source available as an explicit diagnostic
    source.
  - Update focused smoke tests for residual improvement, Schur acceptance, and
    gauge-canonical supported DOF recovery with the stopped latent.
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

- [x] Switch the default geometry update source to stopped reconstruction.
- [x] Make the deterministic smoke geometry observable with stopped latent.
- [x] Update focused smoke expectations and artifact source checks.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the stopped-latent Schur slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_continuation.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_continuation.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_continuation.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 14 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep the fixed synthetic truth source available only as an explicit
  diagnostic source.
- The smoke fixture now uses wider angular residual variation with smaller
  detector shifts so the stopped latent leaves an observable Schur residual
  while still recovering the supported DOFs.
- The smoke32 preview Schur budget is 8 LM iterations; this is the smallest
  observed budget that accepts a stopped-latent update deterministically.

### Risks

- Risk: the stopped latent can absorb geometry if the smoke fixture is too
  weakly observable.
- Mitigation: use a deterministic supported-DOF corruption that produces a
  measurable stopped-latent Schur residual and recovery.

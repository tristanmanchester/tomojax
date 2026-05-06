# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Alternating solver and continuation
- Goal: add robust residual-scale estimation to Phase 7 continuation.

### Scope

- In scope:
  - Add a typed robust residual-scale estimator in `tomojax.forward`.
  - Use per-level robust residual scale in the Phase 7 smoke path with the
    schedule sigma as a stability floor.
  - Record estimated/effective residual sigma in level summaries and artifacts.
  - Add focused forward and smoke tests.
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

- [x] Add public robust residual-scale estimator.
- [x] Thread estimated/effective sigma through Phase 7 level execution.
- [x] Record sigma in summaries/artifacts.
- [x] Add focused forward and smoke tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the robust-scale continuation slice.

### Validation

- `uv run ruff format src/tomojax/forward/_residuals.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py src/tomojax/align/_alternating.py tests/test_forward_reference.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/forward/_residuals.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py src/tomojax/align/_alternating.py tests/test_forward_reference.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/forward/_residuals.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py src/tomojax/align/_alternating.py tests/test_forward_reference.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_forward_reference.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 16 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Use the schedule residual sigma as a lower bound because the sparse 32^3
  smoke residual has a very small MAD scale.

### Risks

- Risk: the current smoke fixture is sparse enough that robust scale estimation
  mostly records rather than changes behavior.
- Mitigation: use it as a floor-aware hook now and let noisier benchmark
  ingestion exercise scale increases later.

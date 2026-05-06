# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Alternating solver and continuation
- Goal: prove the first real Schur geometry update inside the deterministic
  32^3 Phase 7 smoke loop.

### Scope

- In scope:
  - Record corrupted-initial versus true-geometry supported DOF recovery in the
    smoke verification payload.
  - Add a focused deterministic 32^3 smoke assertion that runs the supported
    joint Schur LM geometry update with the synthetic truth volume.
  - Verify projection residual improvement, accepted Schur diagnostics, and
    gauge-canonical supported DOF recovery artifacts.
- Out of scope:
  - Artifact-shape polishing beyond fields needed to prove the update.
  - Changing the stopped-reconstruction default geometry-update source.
  - Nuisance model changes.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add initial-vs-final supported DOF recovery fields to verification.
- [x] Add focused Schur-in-loop smoke test with corrupted initial geometry.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Schur-in-loop recovery evidence slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 10 tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 5 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep the default smoke on the stopped-reconstruction volume source; use the
  existing fixed-synthetic-truth source only for the focused Schur recovery
  vertical-slice assertion.

### Risks

- Risk: the stopped-reconstruction default still limits DOF recovery tightness.
- Mitigation: this slice records that evidence separately and uses the
  truth-volume smoke only to prove the Schur update mechanics and artifacts.

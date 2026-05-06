# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 nuisance models and weak DOF handling
- Goal: record fitted nuisance estimates in Schur diagnostics.

### Scope

- In scope:
  - Include fitted gain/offset estimates in joint Schur diagnostics when
    enabled.
  - Include fitted background estimates in joint Schur diagnostics when enabled.
  - Preserve existing geometry-update behavior and artifact schemas.
- Out of scope:
  - New nuisance solver blocks.
  - Automatic weak-DOF activation changes.
  - Broader failure-classifier policy changes.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` with public `tomojax.nuisance` payloads.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add nuisance estimate payloads to Schur diagnostics.
- [x] Add focused diagnostics tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the nuisance diagnostics slice.

### Validation

- `uv run ruff format src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed: 8 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Diagnostics record the nuisance estimate for the current accepted parameter
  state rather than each rejected candidate.

### Risks

- Risk: diagnostics could imply nuisance fitting ran when it was disabled.
- Mitigation: keep payloads `None` unless the matching fit flag is enabled.

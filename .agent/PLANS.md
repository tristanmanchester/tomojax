# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Alternating solver and continuation
- Goal: use Phase 7 continuation prior strength in Schur geometry updates.

### Scope

- In scope:
  - Add a weak quadratic parameter prior to the joint Schur LM solver.
  - Drive that prior from `ContinuationLevel.prior_strength`.
  - Record prior strength in Schur diagnostics and Phase 7 artifacts.
  - Add focused Schur and smoke tests.
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

- [x] Add Schur parameter prior config.
- [x] Include the prior in Schur residuals and losses.
- [x] Thread continuation prior strength into Phase 7 Schur updates.
- [x] Record prior strength in diagnostics/artifacts.
- [x] Add focused Schur and smoke tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the prior-strength slice.

### Validation

- `uv run ruff format src/tomojax/align/_joint_schur_lm.py src/tomojax/align/_alternating.py tests/test_joint_schur_lm.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/_alternating.py tests/test_joint_schur_lm.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/_alternating.py tests/test_joint_schur_lm.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 12 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Use a zero-centered prior around the current Schur solve's initial parameter
  vector. This makes the continuation prior a weak step regulariser without
  adding metadata priors yet.

### Risks

- Risk: this is a simple local parameter prior, not the full metadata/smoothness
  prior family from the design docs.
- Mitigation: keep it explicit in diagnostics and evolve it once nuisance and
  metadata priors exist.

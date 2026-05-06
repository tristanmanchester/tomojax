# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: add opt-in gain/offset variable projection to Schur geometry residuals.

### Scope

- In scope:
  - Add an opt-in Schur config flag for fitting per-view gain/offset nuisance.
  - Apply the nuisance model when evaluating Schur residuals and losses.
  - Record whether gain/offset fitting was enabled in Schur diagnostics.
  - Add a focused synthetic test showing gain drift can be explained without a
    fake geometry update when geometry is already correct.
- Out of scope:
  - Default-enabling nuisance fitting in Phase 7 smoke.
  - Background fields.
  - Weak DOF auto-activation rules.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`, consuming public `tomojax.nuisance`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/03_repo_layout.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Add Schur gain/offset nuisance config and diagnostics.
- [x] Apply fitted nuisance in Schur residual/loss evaluation.
- [x] Update align README dependency and invariant notes.
- [x] Add focused Schur+nuisance test.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Schur nuisance slice.

### Validation

- `uv run ruff format src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py tests/test_nuisance_gain_offset.py -q`
  passed: 11 tests.
- `just imports` passed.
- Note: an earlier `ruff format` invocation mistakenly included
  `src/tomojax/align/README.md` and failed because Ruff does not parse
  Markdown; the corrected Python-file command passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep gain/offset fitting opt-in for this slice so the existing Phase 7 smoke
  behavior remains unchanged while the Schur path gets a real nuisance hook.

### Risks

- Risk: the nuisance fit is recomputed inside finite-difference residual
  evaluation and is not yet optimized for speed.
- Mitigation: this is the JAX reference correctness path; backend acceleration
  is a later phase.

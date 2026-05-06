# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: add opt-in background nuisance fitting to Schur residuals.

### Scope

- In scope:
  - Add an opt-in Schur config flag for background offset fitting.
  - Apply the fitted background model when evaluating Schur residuals and losses.
  - Record background fitting in Schur diagnostics/artifacts.
  - Add a focused synthetic Schur test for background drift without fake geometry.
- Out of scope:
  - Default-enabling background fitting in Phase 7 smoke.
  - CLI/alternating plumbing for the background flag.
  - Stripe/ring bias fields.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`, consuming public `tomojax.nuisance`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add Schur background nuisance config and diagnostics.
- [x] Apply fitted background in Schur residual/loss evaluation.
- [x] Add focused Schur+background test.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Schur background nuisance slice.

### Validation

- `uv run ruff format src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py tests/test_nuisance_background.py -q`
  passed: 12 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep background fitting opt-in and solver-local for this slice, mirroring the
  earlier gain/offset integration.

### Risks

- Risk: background fitting is recomputed in finite-difference residuals and is
  not performance-optimized.
- Mitigation: this remains the correctness-first JAX reference path.

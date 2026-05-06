# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove the `_run_alignment_step` complexity blocker with
  behavior-preserving optimizer and bookkeeping helpers.

### Scope

- In scope:
  - Extract GN alignment-step handling into a private helper.
  - Extract final gauge/loss bookkeeping into private helpers.
  - Preserve optimizer behavior and existing `OuterStat` keys.
  - Run focused Ruff checks and targeted alignment tests.
- Out of scope:
  - Numerical algorithm changes.
  - `_build_pose_objective_bundle` or top-level `align` decomposition.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Extract GN alignment step helper.
- [x] Extract final gauge/loss bookkeeping helpers.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports three
  `_pose_stage.py` complexity findings: `_build_pose_objective_bundle` and
  top-level `align`.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_optimizers.py -q` passed: 10 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k 'gn or gd or lbfgs or smooth_pose_model or pose_model'`
  aborted in the existing JAX/Optax L-BFGS chunking path
  (`test_align_smooth_pose_model_keeps_frozen_dofs`), matching the previously
  observed native L-BFGS instability.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; `_run_alignment_step` is no longer a
  complexity blocker. The first remaining failures are three `_pose_stage.py`
  PLR0912/PLR0915 complexity findings, followed by legacy import/type-alias
  cleanup in `_profiles.py` and `_reconstruction_stage.py`. Formatter churn
  from this command was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep the existing GD and L-BFGS helpers and add matching GN/core
  helpers rather than changing the optimizer dispatch contract.
- Decision: isolate final gauge/loss bookkeeping in private helpers while
  preserving the existing `OuterStat` keys and loss reuse rule for GN.
- Deviation: skipped L-BFGS integration validation after the native JAX/Optax
  abort; non-LBFGS alignment and optimizer unit tests passed.

### Risks

- Risk: optimizer-step extraction could subtly change when losses or gauge stats
  are evaluated.
- Mitigation: preserve existing call order, return the same values, and run
  targeted GN/GD/L-BFGS alignment tests where practical.
- Proposed next fix for `just check`: decompose `_build_pose_objective_bundle`
  before the larger top-level `align` split.

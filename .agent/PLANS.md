# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove the `_build_pose_objective_bundle` complexity blocker with
  behavior-preserving objective helper extraction.

### Scope

- In scope:
  - Extract chunk/mask helpers for pose objective evaluation.
  - Extract manual loss/gradient and GN-update builders.
  - Preserve differentiable objective behavior and `PoseObjectiveBundle` API.
  - Run focused Ruff checks and targeted alignment tests.
- Out of scope:
  - Numerical algorithm changes.
  - Top-level `align` decomposition.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Extract pose objective chunk/mask helpers.
- [x] Extract manual loss/gradient builder.
- [x] Extract GN update builder.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports two
  `_pose_stage.py` complexity findings, both on top-level `align`.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_optimizers.py -q` passed: 10 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; `_build_pose_objective_bundle` is no
  longer a complexity blocker. The first remaining failures are top-level
  `align` PLR0912/PLR0915 findings, followed by legacy import/type-alias
  cleanup in `_profiles.py` and `_reconstruction_stage.py`. Formatter churn
  from this command was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: introduce `_PoseObjectiveContext` to keep the extracted JAX helper
  signatures small while preserving the same captured runtime/config values.
- Decision: keep smoothness loss and gradient handling as explicit private
  helpers shared by align-loss, manual-gradient, and GN-update paths.
- Deviation: none from the cleanup scope.

### Risks

- Risk: objective helper extraction could change JAX closure capture or tracing
  behavior.
- Mitigation: keep helpers private, pass a frozen context object, and run
  targeted GN/GD alignment tests.
- Proposed next fix for `just check`: split the top-level `align` orchestration
  loop into setup, per-outer, and final-info helpers.

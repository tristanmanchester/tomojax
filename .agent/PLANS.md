# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove the top-level `align` complexity blocker with
  behavior-preserving orchestration helper extraction.

### Scope

- In scope:
  - Extract alignment loop state/checkpoint handling.
  - Extract per-outer iteration, observer, early-stop, completion logging, and
    final-info assembly helpers.
  - Preserve the public `align` API and `AlignInfo` payload.
  - Run focused Ruff checks and targeted alignment tests.
- Out of scope:
  - Numerical algorithm changes.
  - Reconstruction, objective, or optimizer formula changes.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Extract align loop state and checkpoint helper.
- [x] Extract per-outer iteration helper.
- [x] Extract observer, early-stop, and completion/final-info helpers.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` passed.
- `uv run ruff format src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_optimizers.py -q` passed: 10 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; top-level `align` is no longer a
  local complexity blocker. The first remaining failures are legacy
  import/type-alias findings in `_profiles.py`, `_quality_policy.py`,
  `_reconstruction_stage.py`, `_results.py`, and `_setup_stage.py`, followed
  by broader repository lint backlog. Formatter churn from this command was
  reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: use a private `_AlignLoopState` to preserve mutable loop behavior
  without spreading resume/checkpoint state through the top-level function.
- Decision: keep setup, runtime, and objective construction in `align` for now
  and extract only the loop, step-context, observer, early-stop, completion,
  and final-info responsibilities needed to clear this complexity blocker.
- Deviation: none from the cleanup scope.

### Risks

- Risk: stateful loop extraction could change resume/checkpoint or early-stop
  behavior.
- Mitigation: keep mutation in a private loop-state object and run targeted
  GN/GD alignment tests plus import checks.
- Proposed next fix for `just check`: clean the legacy import/type-alias
  findings in `_profiles.py`, then continue through `_quality_policy.py` and
  `_reconstruction_stage.py`.

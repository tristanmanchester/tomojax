# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove the `_align_summary_parts` complexity blocker with a
  behavior-preserving summary-formatting split.

### Scope

- In scope:
  - Split `_align_summary_parts` into step-kind and loss summary helpers.
  - Preserve compact and verbose alignment log text.
  - Run focused Ruff checks and targeted summary/alignment tests.
- Out of scope:
  - Numerical alignment behavior.
  - Larger `_run_alignment_step`, `_build_pose_objective_bundle`, or `align`
    decomposition.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Split alignment step summary formatting.
- [x] Split alignment loss summary formatting.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports five
  `_pose_stage.py` complexity findings: `_build_pose_objective_bundle`,
  `_run_alignment_step`, and `align`.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k 'log_summary or log_compact or smooth_pose_model or pose_model'`
  passed: 9 tests, 43 deselected.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; `_align_summary_parts` is no longer a
  complexity blocker. The first remaining failures are five `_pose_stage.py`
  PLR0912/PLR0915 complexity findings, followed by legacy import/type-alias
  cleanup in `_profiles.py` and `_reconstruction_stage.py`. Formatter churn
  from this command was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep summary formatting in `_pose_stage.py` because it is private
  presentation logic for this pipeline.
- Decision: split by step kind (`gn`, `lbfgs`, `gd`) and loss formatting rather
  than introducing a new formatter object.
- Deviation: none from the cleanup scope.

### Risks

- Risk: summary formatting has compact/verbose branches with subtle output
  differences.
- Mitigation: preserve the existing string construction inside small helpers
  and run log-summary-focused tests.
- Proposed next fix for `just check`: split `_run_alignment_step` into
  optimizer-kind helpers before tackling `_build_pose_objective_bundle`.

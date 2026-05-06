# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: reduce the next `just check` Ruff blocker in
  `src/tomojax/align/_pose_stage.py` without weakening checks.

### Scope

- In scope:
  - Convert parent-relative imports in `_pose_stage.py` to absolute imports.
  - Move type-only geometry imports behind `TYPE_CHECKING`.
  - Add local annotations for the nested objective helpers Ruff currently
    reports.
  - Apply simple local autofixable cleanup in the same file.
  - Run focused `_pose_stage.py` static checks and targeted alignment tests.
- Out of scope:
  - Repository-wide legacy Ruff cleanup beyond `_pose_stage.py`.
  - Large behavior refactors of the alignment pipeline.
  - Full complexity decomposition unless it is required for this narrow slice.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Update `_pose_stage.py` imports.
- [x] Add local nested-helper annotations.
- [x] Apply simple local cleanup.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports only seven
  `_pose_stage.py` complexity findings: `_build_pose_objective_bundle`,
  `_align_summary_parts`, `_run_alignment_step`, and `align`.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_chunking.py -q -k 'not lbfgs'` passed:
  24 tests, 5 deselected.
- `uv run pytest tests/test_align_quick.py -q -k 'gn or smooth_pose_model or pose_model'`
  passed: 23 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; the `_pose_stage.py` import,
  type-only import, annotation, lambda, and line-length findings are gone. The
  first remaining failures are `_pose_stage.py` PLR0912/PLR0915 complexity
  findings, followed by legacy import/type-alias cleanup in `_profiles.py` and
  `_reconstruction_stage.py`. Formatter churn from this command was reverted
  outside this slice.
- `uv run basedpyright src/tomojax/align/_pose_stage.py` was not a useful
  focused gate: the file has a broad pre-existing basedpyright backlog
  including private intra-module imports, JAX unknowns, and typed-dict/stat
  narrowing issues.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: use absolute imports only for runtime dependencies and keep
  geometry shape classes type-only under `TYPE_CHECKING`.
- Decision: add broad `jnp.ndarray` annotations to nested JAX helpers without
  changing traced numerical logic.
- Deviation: did not split the large `_pose_stage.py` functions yet; this slice
  only removes the low-risk Ruff findings before the PLR complexity work.

### Risks

- Risk: `_pose_stage.py` contains JAX-traced nested functions where annotations
  must not change runtime behavior.
- Mitigation: use broad `jnp.ndarray` and tuple return annotations only; avoid
  changing traced control flow or numerical logic.
- Proposed next fix for `just check`: decompose `_align_summary_parts` and then
  `_run_alignment_step`/`_build_pose_objective_bundle` in separate small slices
  before continuing to `_profiles.py`.

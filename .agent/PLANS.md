# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove `_stage_loop.py` import, annotation, lambda-binding, and
  unused-resume Ruff blockers before the larger orchestration split.

### Scope

- In scope:
  - Replace parent-relative imports with absolute imports.
  - Move annotation-only imports behind `TYPE_CHECKING`.
  - Add missing annotations for stage observer/checkpoint helpers.
  - Bind phase-correlation lambda loop variables.
  - Remove unused resume-stage locals.
  - Run focused Ruff checks and multires/setup tests.
- Out of scope:
  - Alignment algorithm changes.
  - Large `align_multires` and `_run_multires_level_stages` decomposition.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Clean `_stage_loop.py` imports.
- [x] Add local helper annotations.
- [x] Fix lambda binding and unused resume locals.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_stage_loop.py` now reports only the
  remaining planned complexity blockers in `_run_multires_level_stages` and
  `align_multires`.
- `uv run ruff format src/tomojax/align/_stage_loop.py` passed.
- `uv run pytest tests/test_multires.py tests/test_bilevel_setup_alignment.py tests/test_align_checkpoint.py -q`
  passed: 43 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; the first remaining failures are
  `_stage_loop.py` PLR0915/PLR0912 complexity findings, followed by geometry
  module doc/import findings and later modules/tests. Formatter churn from
  this command was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: move `bin_projections`, scale helpers, phase correlation, and core
  imports to module scope with absolute imports instead of local parent-relative
  imports.
- Decision: keep observer/result types behind `TYPE_CHECKING`; runtime uses
  only the normalizer and adapter functions.
- Decision: replace the phase-correlation lambda with a local function using
  default-bound loop values to preserve per-level inputs under `jax.vmap`.
- Deviation: none from the cleanup scope.

### Risks

- Risk: lambda/default binding changes touch the optional translation seeding
  path.
- Mitigation: preserve the existing call signature and run focused multires
  tests including setup path coverage.
- Proposed next fix for `just check`: split `_run_multires_level_stages` and
  `align_multires` orchestration complexity.

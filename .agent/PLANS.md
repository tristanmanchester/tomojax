# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove `_setup_stage.py` import and missing-annotation Ruff blockers
  with behavior-preserving typing cleanup.

### Scope

- In scope:
  - Replace parent-relative imports with absolute imports.
  - Move annotation-only imports behind `TYPE_CHECKING`.
  - Add missing annotations for fold arrays, loss adapter, and loss specs.
  - Preserve setup validation objective behavior.
  - Run focused Ruff checks and setup/alignment tests.
- Out of scope:
  - Alignment algorithm changes.
  - Setup optimisation formula changes.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Clean `_setup_stage.py` imports.
- [x] Add missing annotations.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_setup_stage.py tests/test_bilevel_setup_alignment.py`
  passed.
- `uv run ruff format src/tomojax/align/_setup_stage.py tests/test_bilevel_setup_alignment.py`
  passed.
- `uv run pytest tests/test_bilevel_setup_alignment.py tests/test_align_profiles.py -q`
  passed: 12 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; `_setup_stage.py` is no longer in the
  failure list. The first remaining failures are import/type annotation,
  complexity, loop-binding, and unused-variable findings in `_stage_loop.py`,
  followed by geometry module doc/import findings and later modules/tests.
  Formatter churn from this command was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: move only annotation-only names to `TYPE_CHECKING`; runtime setup
  execution imports remain runtime imports.
- Decision: update the setup-stage test's manual `ResolvedAlignmentStage`
  construction to the current schedule contract instead of bypassing the test.
- Decision: clean touched-file test lint exposed by focused Ruff so this slice
  does not leave known lint in modified files.
- Deviation: none from the cleanup scope.

### Risks

- Risk: typing changes could accidentally move runtime dependencies behind
  `TYPE_CHECKING`.
- Mitigation: keep construction/execution imports at runtime, move only
  annotation-only names, and run setup/alignment tests.
- Proposed next fix for `just check`: continue into `_stage_loop.py`.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove `_profiles.py` import/type-alias Ruff blockers with
  behavior-preserving typing cleanup.

### Scope

- In scope:
  - Convert `_profiles.py` aliases to PEP 695 `type` aliases.
  - Move annotation-only imports behind `TYPE_CHECKING`.
  - Replace parent-relative imports with absolute type-checking imports.
  - Run focused Ruff checks and profile-related tests.
- Out of scope:
  - Alignment algorithm changes.
  - Reconstruction-stage cleanup.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Convert `_profiles.py` aliases.
- [x] Move annotation-only imports.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_profiles.py` passed.
- `uv run ruff format src/tomojax/align/_profiles.py` passed.
- `uv run pytest tests/test_align_profiles.py -q` passed: 6 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; `_profiles.py` is no longer in the
  failure list. The first remaining failures are `_quality_policy.py` UP040,
  then legacy import/type/complexity findings in `_reconstruction_stage.py`,
  `_results.py`, `_setup_stage.py`, and later modules/tests. Formatter churn
  from this command was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep `ProjectorBackend`, `Regulariser`, and `Mapping` as
  annotation-only imports under `TYPE_CHECKING`.
- Decision: use string-based `cast` targets for annotation-only types that are
  no longer imported at runtime.
- Deviation: none from the cleanup scope.

### Risks

- Risk: moving imports under `TYPE_CHECKING` could break runtime casts if the
  names are still evaluated at runtime.
- Mitigation: use string-based `cast` targets for annotation-only types and
  run focused profile tests.
- Proposed next fix for `just check`: continue through `_quality_policy.py`
  and `_reconstruction_stage.py`.

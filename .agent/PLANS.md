# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove `_quality_policy.py` type-alias Ruff blocker with
  behavior-preserving typing cleanup.

### Scope

- In scope:
  - Convert `_quality_policy.py` alias to a PEP 695 `type` alias.
  - Keep quality policy behavior unchanged.
  - Run focused Ruff checks and quality/profile tests.
- Out of scope:
  - Alignment algorithm changes.
  - Reconstruction-stage cleanup.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Convert `_quality_policy.py` alias.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_quality_policy.py` passed.
- `uv run ruff format src/tomojax/align/_quality_policy.py` passed.
- `uv run pytest tests/test_align_profiles.py -q` passed: 6 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; `_quality_policy.py` is no longer in
  the failure list. The first remaining failures are legacy import/type,
  return-annotation, and complexity findings in `_reconstruction_stage.py`,
  followed by `_results.py`, `_setup_stage.py`, and later modules/tests.
  Formatter churn from this command was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep the quality tier alias public name unchanged while switching
  its declaration to PEP 695 syntax.
- Decision: use Ruff's file-local fix for cast quoting and import ordering.
- Deviation: none from the cleanup scope.

### Risks

- Risk: minimal; the alias conversion should be type-only.
- Mitigation: run focused Ruff and profile tests that exercise quality tier
  normalization through profile configuration.
- Proposed next fix for `just check`: continue into `_reconstruction_stage.py`.

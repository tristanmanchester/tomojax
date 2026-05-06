# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove `_results.py` type-only import Ruff blockers with
  behavior-preserving typing cleanup.

### Scope

- In scope:
  - Move annotation-only `jax.numpy`, observer, and schedule imports behind
    `TYPE_CHECKING`.
  - Use `collections.abc` for runtime collection protocols.
  - Preserve result dataclass, callback, and `TypedDict` public APIs.
  - Run focused Ruff checks and result/alignment tests.
- Out of scope:
  - Alignment algorithm changes.
  - Result schema changes.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Move `_results.py` annotation-only imports.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_results.py` passed.
- `uv run ruff format src/tomojax/align/_results.py` passed.
- `uv run pytest tests/test_align_checkpoint.py tests/test_align_profiles.py -q`
  passed: 16 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; `_results.py` is no longer in the
  failure list. The first remaining failures are import/type annotation
  findings in `_setup_stage.py`, followed by `_stage_loop.py` and later
  modules/tests. Formatter churn from this command was reverted outside this
  slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep `TypedDict` imported at runtime because the result schemas
  subclass it.
- Decision: accept Ruff's local fixes for `__all__` sorting and direct
  `cfg.spdhg_seed` access while preserving the existing required config
  contract.
- Deviation: none from the cleanup scope.

### Risks

- Risk: moving result annotation imports behind `TYPE_CHECKING` could expose a
  runtime dependency if any code introspects annotations without postponed
  evaluation.
- Mitigation: rely on `from __future__ import annotations` and run focused
  tests that construct resume states and use result helpers.
- Proposed next fix for `just check`: continue into `_setup_stage.py`.

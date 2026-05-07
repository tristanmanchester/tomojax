# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 unsupported DOF classification
- Goal: make benchmark manifest criteria for unsupported DOFs explicitly
  `unsupported_dof_not_evaluated` instead of treating them as supported recovery
  gates.

### Scope

- In scope:
  - Update benchmark manifest criterion evaluation reasons for unsupported
    criteria.
  - Add focused CLI artifact assertions for unsupported axis/roll criteria.
  - Run focused validation and import checks.
- Out of scope:
  - Implementing additional DOFs or rerunning the full five-case benchmark.
  - Legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Mark unsupported manifest criteria as `unsupported_dof_not_evaluated`.
- [x] Add focused artifact test assertions.
- [x] Run focused validation and `just imports`.
- [x] Update docs and commit the unsupported-DOF classification slice.

### Validation

- `uv run ruff format ...` passed for touched benchmark artifact/test files.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py
  tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py
  tests/test_align_auto_cli.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest` on three focused align-auto artifact
  tests passed: 3 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Unsupported criteria still count toward `not_evaluated` in the manifest
  summary, but their reason is now explicit and machine-readable.

### Risks

- Risk: five-case runs generated before this slice still contain the older
  generic reason string.
- Mitigation: rerun the five-case comparison only after supported-only
  production-like recovery is ready to judge.

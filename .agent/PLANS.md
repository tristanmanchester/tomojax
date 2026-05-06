# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: add a residual-structure gate for `nuisance_unmodelled` failure reporting.

### Scope

- In scope:
  - Add a residual-structure gate to `failure_report.json`.
  - Report `nuisance_unmodelled` warnings when detector-column or per-view
    residual structure is strong.
  - Keep default smoke passing.
  - Add focused tests for passing and warning cases.
- Out of scope:
  - Full stripe/background model fitting.
  - Making nuisance warnings fail the run.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add residual structure summary helper.
- [x] Add nuisance gate and warning mapping.
- [x] Add focused smoke and unit-style tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the nuisance failure gate slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/verify/_residual_structure.py src/tomojax/verify/api.py src/tomojax/verify/__init__.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/verify/_residual_structure.py src/tomojax/verify/api.py src/tomojax/verify/__init__.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/verify/_residual_structure.py src/tomojax/verify/api.py src/tomojax/verify/__init__.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 8 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- The gate is warning-only because Phase 8 nuisance models are still incomplete.

### Risks

- Risk: the structure heuristic is simple and may not catch all nuisance modes.
- Mitigation: expose the metrics in failure evidence and evolve the gate with
  background/stripe models.

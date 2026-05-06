# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 nuisance models and weak DOF handling
- Goal: replace weak-DOF validation-improvement placeholder evidence with real
  Schur improvement evidence in the smoke observability policy.

### Scope

- In scope:
  - Populate weak-DOF policy evidence from Schur actual reduction when
    diagnostics exist.
  - Keep inactive/frozen DOFs explicit.
  - Update focused observability assertions.
- Out of scope:
  - Stripe/ring bias fields.
  - Larger 128^3 benchmark runtime.
  - Detector-shift volume gauge correction.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add Schur validation-improvement evidence to weak DOF decisions.
- [x] Update focused observability assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the weak-DOF evidence slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_alternating_solver_smoke.py`
  passed: 3 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Do not change weak-DOF decisions in this slice; add evidence only.

### Risks

- Risk: Schur actual reduction is optimisation evidence, not a held-out
  validation metric.
- Mitigation: label it as `schur_actual_reduction` and keep held-out validation
  for a later policy slice.

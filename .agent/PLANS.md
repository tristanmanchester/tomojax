# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 alternating solver and continuation vertical slice
- Goal: split the alternating smoke implementation into cohesive private
  align-owned modules before adding synthetic benchmark ingestion.

### Scope

- In scope:
  - Preserve the public `tomojax.align` API for alternating smoke types and
    runner entrypoints.
  - Split `src/tomojax/align/_alternating.py` into private modules for smoke
    config/result helpers, held-out checks, verification/report payloads,
    artifact writing, and orchestration.
  - Keep cross-file imports inside the `tomojax.align` private boundary.
  - Run focused smoke/CLI tests and `just imports`.
- Out of scope:
  - Synthetic benchmark ingestion beyond preserving existing sidecar readback
    behavior.
  - Stripe/ring bias fields.
  - Changing solver behavior.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Move smoke config/result dataclasses to a private align module.
- [x] Move held-out residual checks to a private align module.
- [x] Move verification/report payload builders to a private align module.
- [x] Move artifact writing to a private align module.
- [x] Leave `_alternating.py` as orchestration plus solver loop.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the align-module split slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_heldout.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py`
  passed: 5 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_heldout.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_heldout.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 14 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- This is a structural cleanup only. Do not change solver behavior in this
  slice.

### Risks

- Risk: moving private functions can accidentally widen public API or introduce
  cross-boundary private imports.
- Mitigation: only import private modules from inside `tomojax.align`; validate
  with focused tests and `just imports`.

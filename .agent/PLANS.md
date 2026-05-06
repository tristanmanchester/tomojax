# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 alternating solver and continuation vertical slice
- Goal: add an executable stopped-reconstruction sidecar contract and record
  the remaining reconstruction-gauge recovery gap.

### Scope

- In scope:
  - Quantify the current sidecar-backed 32^3 smoke run with the default
    `stopped_reconstruction` update volume source.
  - Preserve the fixed-truth sidecar Schur test as an isolating solver check.
  - Add focused assertions that the stopped-reconstruction sidecar path uses the
    real Schur update and improves projection residual/supported DOFs.
  - Record that absolute detector-shift recovery still needs reconstruction
    gauge handling.
- Out of scope:
  - Stripe/ring bias fields.
  - Larger 128^3 benchmark runtime.
  - New placeholder artifact/report polish.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Identify why sidecar stopped-reconstruction recovery fails today.
- [x] Add focused stopped-reconstruction sidecar assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the stopped-reconstruction sidecar slice.

### Validation

- `uv run ruff format tests/test_alternating_solver_smoke.py`
  passed: 1 file left unchanged.
- `uv run ruff check tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed.
- `just check` failed in legacy Ruff cleanup before typecheck/tests:
  - `uv run ruff format src tests tools` reformatted 70 legacy files.
  - `uv run ruff check --fix src tests tools` fixed 320 issues and left 1364
    Ruff issues, starting in transitional `src/tomojax/align/model/schedules.py`
    and `src/tomojax/align/model/state.py`.
  - The unrelated formatter churn from this broad command was reverted.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Do not relax the fixed-truth recovery gate. The stopped-reconstruction test
  records the current real-loop behavior separately until reconstruction gauge
  handling is implemented.

### Risks

- Finding: the current geometry-aware backprojection bakes detector shift into
  the stopped volume. Schur improves residual, but absolute det_u recovery
  remains outside the smoke tolerance.
- Mitigation: keep the limitation executable and explicit; next slice should
  address reconstruction/volume gauge handling rather than weakening recovery
  checks.

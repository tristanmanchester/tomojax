# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 alternating solver and continuation vertical slice
- Goal: record reconstruction/volume gauge diagnostics for sidecar stopped
  reconstruction runs.

### Scope

- In scope:
  - Add compact verification metrics that compare initial/final stopped-volume
    projection losses under true and corrupted geometry for sidecar runs.
  - Expose whether the stopped volume is closer to the corrupted gauge than the
    true gauge.
  - Add focused assertions for the current sidecar stopped-reconstruction gap.
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

- [x] Add stopped-volume gauge diagnostic payload.
- [x] Add focused assertions for sidecar stopped-volume gauge evidence.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the stopped-volume gauge diagnostic slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed: 3 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 16 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Do not relax recovery tolerances. This slice is diagnostic evidence for the
  next reconstruction-gauge fix.

### Risks

- Risk: diagnostics can turn into placeholder artifact polish.
- Mitigation: add only metrics used by the focused sidecar regression test and
  keep the next implementation target tied to those numbers.

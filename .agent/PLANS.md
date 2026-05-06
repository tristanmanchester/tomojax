# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: preserve synthetic benchmark manifest pass criteria in smoke-run
  benchmark artifacts.

### Scope

- In scope:
  - Include sidecar `recovery_tolerances` in synthetic dataset readback.
  - Thread those manifest criteria into `benchmark_result.json`.
  - Render them in `benchmark_report.md`.
  - Add focused CLI assertions for the existing-sidecar path.
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

- [x] Add manifest pass criteria to synthetic sidecar readback.
- [x] Thread pass criteria into benchmark result/report.
- [x] Add focused pass-criteria assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the benchmark pass-criteria slice.

### Validation

- `uv run ruff format src/tomojax/cli/align_auto.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed: 3 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/cli/align_auto.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/cli/align_auto.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 7 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep benchmark manifest criteria separate from current smoke acceptance
  tolerances; do not change solver pass/fail behavior in this slice.

### Risks

- Risk: manifest pass criteria can be mistaken for active smoke gates.
- Mitigation: label them as benchmark manifest criteria in result/report.

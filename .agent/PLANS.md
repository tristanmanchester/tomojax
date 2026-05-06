# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: allow the deterministic align-auto smoke command to ingest an existing
  generated synthetic benchmark sidecar directory.

### Scope

- In scope:
  - Add an explicit existing-sidecar input option to `tomojax.cli.align_auto`.
  - Validate and record sidecar readback before running the solver.
  - Add focused CLI coverage proving projections are loaded from the existing
    benchmark directory.
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

- [x] Add existing-sidecar input option to `align_auto`.
- [x] Add focused CLI ingestion assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the benchmark-ingestion slice.

### Validation

- `uv run ruff format src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed: 2 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 7 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Existing sidecar ingestion should not regenerate or overwrite benchmark
  artifacts.
- Keep the command scoped to 32^3 focused validation unless explicitly running
  a larger benchmark.

### Risks

- Risk: CLI size/views can drift from the supplied sidecars.
- Mitigation: pass configured size/views through the existing sidecar shape
  checks and fail before writing misleading run artifacts.

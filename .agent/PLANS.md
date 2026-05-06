# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: validate optional synthetic benchmark result artifacts in the run
  artifact verifier.

### Scope

- In scope:
  - Load optional `benchmark_result.json` when present.
  - Validate its schema and required benchmark result sections.
  - Add focused verifier coverage for a synthetic benchmark run.
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

- [x] Add optional benchmark-result validation.
- [x] Add focused verifier coverage for benchmark artifacts.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the benchmark verifier slice.

### Validation

- `uv run ruff format src/tomojax/verify/_artifacts.py tests/test_verify_artifacts.py`
  passed: 2 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/verify/_artifacts.py tests/test_verify_artifacts.py`
  passed.
- `uv run basedpyright src/tomojax/verify/_artifacts.py tests/test_verify_artifacts.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_verify_artifacts.py -q`
  passed: 3 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Benchmark result validation is optional and only applies when the artifact is
  present.

### Risks

- Risk: making benchmark artifacts mandatory would break non-benchmark smoke
  runs.
- Mitigation: load and validate `benchmark_result.json` only when it exists.

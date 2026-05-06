# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: expose synthetic benchmark result comparison as a real CLI command.

### Scope

- In scope:
  - Add a command-line entrypoint over the existing benchmark-result ingestion
    helper.
  - Support writing a deterministic markdown comparison report to a requested
    output path.
  - Support stdout preview when no output path is supplied.
- Out of scope:
  - Full current-vs-reimagined protocol runner.
  - New synthetic dataset generation behavior.
  - Larger 128^3 benchmark runtime.
  - Further legacy Ruff cleanup.
- Deep module owner: transitional `tomojax.bench`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add benchmark comparison CLI entrypoint.
- [x] Add focused CLI tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the benchmark comparison CLI slice.

### Validation

- `uv run ruff format src/tomojax/bench/synthetic_results.py tests/test_bench_synthetic_results.py pyproject.toml`
  passed.
- `uv run ruff check src/tomojax/bench/synthetic_results.py tests/test_bench_synthetic_results.py pyproject.toml`
  passed.
- `uv run basedpyright src/tomojax/bench/synthetic_results.py tests/test_bench_synthetic_results.py`
  passed.
- `uv run pytest tests/test_bench_synthetic_results.py -q` passed: 6 tests.
- `just imports` passed.
- `uv run tomojax-synthetic-benchmark-compare --help` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- The CLI stays on the existing `tomojax.bench.synthetic_results` helper rather
  than creating a new benchmark runner.

### Risks

- Risk: transitional `tomojax.bench` is not yet a v2 deep module.
- Mitigation: keep the command narrow and limited to result-artifact ingestion.

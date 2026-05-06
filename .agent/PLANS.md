# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: ingest synthetic benchmark result artifacts into a deterministic
  comparison report.

### Scope

- In scope:
  - Load one or more `benchmark_result.json` artifacts.
  - Validate the synthetic benchmark result schema before comparison.
  - Emit a deterministic markdown comparison table over actual benchmark result
    fields.
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

- [x] Add benchmark-result artifact loading.
- [x] Add deterministic markdown comparison rendering.
- [x] Add focused tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the benchmark-ingestion slice.

### Validation

- `uv run ruff format src/tomojax/bench/synthetic_results.py src/tomojax/bench/__init__.py tests/test_bench_synthetic_results.py`
  passed.
- `uv run ruff check src/tomojax/bench/synthetic_results.py src/tomojax/bench/__init__.py tests/test_bench_synthetic_results.py`
  passed.
- `uv run basedpyright src/tomojax/bench/synthetic_results.py tests/test_bench_synthetic_results.py`
  passed.
- `uv run pytest tests/test_bench_synthetic_results.py -q` passed: 4 tests.
- `uv run pytest tests/test_bench_fitness_imports.py tests/test_bench_synthetic_results.py -q`
  passed: 5 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- The comparison helper operates on committed artifact schema fields only and
  does not infer unsupported benchmark criteria.

### Risks

- Risk: transitional `tomojax.bench` is not yet a v2 deep module.
- Mitigation: keep the ingestion helper private-owned within `tomojax.bench`
  and expose only a narrow typed API.

# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic benchmark ingestion
- Goal: emit a machine-readable benchmark case result for deterministic
  synthetic sidecar smoke runs.

### Scope

- In scope:
  - Add a `benchmark_result.json` artifact for runs with synthetic dataset
    metadata.
  - Include dataset/profile/status, core reconstruction and geometry metrics,
    runtime placeholders, backend provenance, and failure labels.
  - Add focused CLI coverage for an existing-sidecar benchmark result.
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

- [x] Add synthetic benchmark result artifact.
- [x] Add focused benchmark-result assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the benchmark-result slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed: 2 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 7 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Emit a JSON result first; markdown comparison reports and current-vs-v2
  comparators remain later slices.
- Keep runtime metrics deterministic placeholders until real timing is wired
  through the solver.

### Risks

- Risk: benchmark result can imply a full protocol run when it is only one
  focused case.
- Mitigation: label the schema and implementation as
  `reimagined_align_auto_smoke` and include the smoke profile/shape.

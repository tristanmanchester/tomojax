# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 alternating solver and continuation vertical slice
- Goal: ingest generated 32^3 synthetic benchmark sidecars into the alternating
  smoke runner and validate real Schur geometry recovery.

### Scope

- In scope:
  - Load generated synthetic sidecar volume/projections/mask/geometry for the
    deterministic 32^3 smoke run when requested.
  - Use the supported joint Schur LM solver as the geometry update.
  - Start from corrupted geometry that differs from true geometry.
  - Verify projection residual improvement and supported DOF recovery after
    gauge canonicalisation.
  - Record Schur diagnostics and geometry trace artifacts from the real update.
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

- [x] Add sidecar-backed smoke inputs without changing default public API.
- [x] Use corrupted sidecar geometry as initial geometry and true sidecar
  geometry for recovery checks.
- [x] Add focused assertions for residual improvement, supported DOF recovery,
  Schur diagnostics, and geometry trace artifacts.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the synthetic ingestion vertical slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_alternating_inputs.py src/tomojax/align/_alternating_verification.py src/tomojax/datasets/_writer.py tests/test_alternating_solver_smoke.py tests/test_synthetic_datasets.py`
  passed during implementation.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_alternating_inputs.py src/tomojax/align/_alternating_verification.py src/tomojax/datasets/_writer.py tests/test_alternating_solver_smoke.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_alternating_inputs.py src/tomojax/align/_alternating_verification.py src/tomojax/datasets/_writer.py tests/test_alternating_solver_smoke.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py -q`
  passed: 25 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep ingestion scoped to generated 32^3 sidecars and the existing reference
  Schur solver. Do not add another placeholder report layer.

### Risks

- Risk: generated NumPy smoke projections may not match the JAX reference
  projector exactly enough for strict recovery gates.
- Mitigation: use the sidecar consistency payload and deterministic focused
  tests to quantify residual and supported DOF improvement before broadening.

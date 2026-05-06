# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: surface synthetic sidecar consistency in the smoke failure report.

### Scope

- In scope:
  - Add a warning-only `synthetic_sidecar_consistency` failure-report gate.
  - Pass the gate when no synthetic sidecar is present.
  - Use the loader consistency payload when a synthetic sidecar is present.
  - Add focused smoke/CLI assertions.
- Out of scope:
  - Alternating solver ingestion of generated benchmark projections.
  - Stripe/ring bias fields.
  - Changing align-auto defaults.
  - Removing existing manifest-spec geometry artifacts.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add synthetic sidecar consistency gate.
- [x] Add focused smoke/CLI assertions.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [ ] Commit the failure-report sidecar gate slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed: 3 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 14 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep this warning-only and provenance-oriented; do not fail the smoke run on
  sidecar metadata in this slice.

### Risks

- Risk: the gate can be mistaken for solver validation of generated
  projections.
- Mitigation: evidence names the sidecar readback path and remains warning-only.

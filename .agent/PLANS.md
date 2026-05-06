# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 weak DOF handling
- Goal: add separate setup and pose priors for joint Schur LM diagnostics.

### Scope

- In scope:
  - Add optional setup/pose prior strengths to `JointSchurLMConfig`.
  - Preserve existing `parameter_prior_strength` behavior by default.
  - Add focused Schur tests showing stronger pose prior reduces pose drift.
- Out of scope:
  - Adding or changing artifact/report/observability fields.
  - New benchmark ingestion behavior.
  - Changing the default stopped-reconstruction solver path.
  - Solver tuning.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` joint Schur LM solver.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add optional setup/pose prior strengths.
- [x] Add focused Schur prior behavior tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Schur prior slice.

### Validation

- `uv run ruff format src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed: 2 files reformatted.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed: 9 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep `parameter_prior_strength` as the default shared prior; optional
  per-block priors override it only when explicitly set.

### Risks

- Risk: stronger pose priors can bias pose-heavy cases.
- Mitigation: do not change defaults; expose behavior through explicit config
  and focused tests only.

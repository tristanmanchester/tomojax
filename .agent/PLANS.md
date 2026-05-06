# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Alternating solver and continuation
- Goal: use the supported joint Schur LM solver as the real Phase 7 smoke
  geometry update.

### Scope

- In scope:
  - Generate observations from true geometry and start from a corrupted initial
    geometry.
  - Call `solve_joint_schur_lm` inside the deterministic 32^3 continuation
    loop for geometry updates.
  - Verify projection residual improvement and supported DOF recovery after
    gauge canonicalisation.
  - Record Schur diagnostics in geometry trace/artifacts.
- Out of scope:
  - Further legacy Ruff cleanup.
  - GPU/Pallas fast paths.
  - Full production dataset loading through the new command.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Switch smoke observations to true geometry with corrupted initial state.
- [x] Use joint Schur LM for continuation geometry updates.
- [x] Record Schur diagnostics in artifacts/traces.
- [x] Extend focused smoke tests for residual improvement and DOF recovery.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Schur-in-loop slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_joint_schur_lm.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 14 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Use the existing supported joint Schur solver; do not introduce a new
  optimizer implementation in this slice.

### Risks

- Risk: the tiny smoke reconstruction may not be a high-quality latent volume.
- Mitigation: keep the slice deterministic and focused on supported DOF
  recovery/residual improvement in the Schur geometry update.

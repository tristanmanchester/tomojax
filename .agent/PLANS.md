# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Alternating solver and continuation
- Goal: initialise the Phase 7 smoke reconstruction with geometry-aware
  reference backprojection.

### Scope

- In scope:
  - Add a `tomojax.recon` public reference backprojection helper.
  - Use geometry-aware backprojection as the initial volume for the Phase 7
    smoke FISTA path.
  - Keep Schur geometry updates on the fixed synthetic smoke volume until the
    stopped latent is strong enough to drive recovery.
  - Add focused reconstruction and smoke tests.
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

- [x] Add public geometry-aware reference backprojection.
- [x] Use backprojection initialization in the Phase 7 smoke FISTA path.
- [x] Add focused reconstruction and smoke tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the backprojection-init slice.

### Validation

- `uv run ruff format src/tomojax/recon/_reference.py src/tomojax/recon/__init__.py src/tomojax/recon/api.py src/tomojax/align/_alternating.py tests/test_reference_fista.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/recon/_reference.py src/tomojax/recon/__init__.py src/tomojax/recon/api.py src/tomojax/align/_alternating.py tests/test_reference_fista.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/recon/_reference.py src/tomojax/recon/__init__.py src/tomojax/recon/api.py src/tomojax/align/_alternating.py tests/test_reference_fista.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_reference_fista.py tests/test_alternating_solver_smoke.py tests/test_joint_schur_lm.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 17 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep the Schur update source explicit as the fixed synthetic smoke volume for
  now; this slice improves the stopped reconstruction latent only.

### Risks

- Risk: geometry-aware backprojection is a simple reference initializer, not a
  production reconstruction.
- Mitigation: keep it public, typed, tested, and owned by `tomojax.recon` as a
  deterministic reference primitive.

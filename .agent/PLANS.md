# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 Schur block-wise trust
- Goal: prevent aggregate pose updates from shrinking valid setup updates in
  the supported fixed-truth joint path.

### Scope

- In scope:
  - Replace global Schur trust scaling with separate setup and pose scaling.
  - Preserve existing diagnostics compatibility while recording enough evidence
    to show setup was not clipped by pose.
  - Add a regression test where a large pose block and small setup block produce
    an unclipped setup step with a clipped pose step.
  - Rerun focused Schur/align tests and the supported-only fixed-truth joint GPU
    benchmark without the effectively hard pose prior.
- Out of scope:
  - Full per-DOF unit trust for every supported/unsupported future DOF.
  - Stopped-reconstruction classification.
  - Report-field expansion beyond diagnostics needed for this fix.
  - Legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Implement separate setup/pose trust scaling in Schur step construction.
- [x] Add block-wise trust regression test.
- [x] Run focused validation and `just imports`.
- [x] Rerun supported-only fixed-truth joint GPU benchmark without hard pose
  prior.
- [x] Update docs with result/blocker.
- [x] Commit the block-wise trust slice.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed: 11 tests.
- `just imports` passed.
- GPU no-hard-prior fixed-truth joint run still failed:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_block_trust/`.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- The hard pose-prior run proves the fixed-truth joint path is recoverable, but
  it is diagnostic. The production-quality next step is trust/gauge handling.
- Block-wise setup/pose trust removes aggregate pose clipping but does not solve
  the remaining setup/pose gauge coupling by itself.

### Risks

- Risk: setup/pose block-wise trust may improve detector recovery but still
  leave theta absorbed into `phi_residual_rad`.
- Mitigation: classify that separately and then stage/freeze `phi` rather than
  broadening scope.

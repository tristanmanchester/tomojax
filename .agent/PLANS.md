# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 Schur residual filtering
- Goal: apply the continuation residual filters inside the actual joint Schur
  residual/Jacobian path and verify low-pass setup updates.

### Scope

- In scope:
  - Add residual-filter schedule configuration to `JointSchurLMConfig`.
  - Thread continuation-level filters into `_run_geometry_updates`.
  - Use filtered residuals consistently for Schur weights, finite-difference
    Jacobian rows, candidate losses, and per-view diagnostics.
  - Add a focused low-pass setup-step regression test.
  - Refresh supported-only fixed-truth and stopped-reconstruction GPU
    diagnostics.
- Out of scope:
  - Reconstruction gauge fixes, benchmark scenario expansion, or additional
    report fields.
  - Legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Thread continuation residual filters into joint Schur.
- [x] Add focused low-pass setup-step test.
- [x] Run focused validation and `just imports`.
- [x] Rerun supported-only fixed-truth/stopped-reconstruction GPU diagnostics.
- [x] Update docs and commit the Schur-filter slice.

### Validation

- `uv run ruff format ...` passed for touched Schur-filter source/test files.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py
  src/tomojax/align/_alternating_geometry_update.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py
  src/tomojax/align/_alternating_geometry_update.py tests/test_joint_schur_lm.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed: 14 tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest` on the two focused align-auto artifact
  contract tests passed: 2 tests.
- `just imports` passed.
- GPU fixed-truth filtered refresh passed:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_pose_prior_1000000_filtered_reporting/`.
- GPU stopped-reconstruction filtered refresh failed as expected:
  `.artifacts/phase8_supported_only_oracle/runs/64_stopped_reconstruction_joint_pose_prior_1000000_filtered_reporting/`.
- Compare artifact:
  `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_filtered_reporting.md`.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Applying continuation filters inside Schur made the fixed-truth strong-prior
  supported-only run essentially exact (`det_u` RMSE about `5.24e-06` px,
  theta RMSE about `5.10e-05` rad).
- Stopped reconstruction still fails without geometry movement and remains
  classified as `reconstruction_absorbed_geometry`.

### Risks

- Risk: final-level diagnostics report `raw` because the last balanced level is
  raw, even though coarser Schur levels now use low-pass/band-pass schedules.
- Mitigation: level summaries continue to record per-level filter kinds; future
  observability can expose iteration-level filter history if needed.
